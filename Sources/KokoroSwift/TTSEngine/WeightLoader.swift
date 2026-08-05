//
//  Kokoro-tts-lib
//
import Foundation
import MLX
import MLXNN

/// Utility class for loading and preprocessing neural network weights.
///
/// WeightLoader handles the loading of model weights from disk and applies necessary
/// transformations to ensure compatibility with the model architecture. This includes:
/// - Filtering out unnecessary weights (e.g., position_ids)
/// - Transposing weight tensors for specific layers
/// - Validating and processing weight shapes
///
/// The class processes weights for different model components:
/// - BERT encoder weights
/// - Predictor (duration and prosody) weights
/// - Text encoder weights
/// - Decoder weights
final class WeightLoader {
  /// WeightLoader is a utility class with only static methods.
  private init() {}

  /// Loads and sanitizes model weights from the specified path.
  /// This method reads the raw model weights and applies component-specific transformations:
  /// - **BERT weights**: Filters out position_ids (not needed for inference)
  /// - **Predictor weights**: Transposes F0 and N projection weights, handles weight_v conditionally
  /// - **Text encoder weights**: Handles weight_v with conditional transposition
  /// - **Decoder weights**: Transposes noise convolution weights and handles weight_v conditionally
  /// - Parameter modelPath: URL to the directory containing model weight files
  /// - Returns: Dictionary mapping weight names to their processed MLXArray tensors
  /// - Note: Uses forced try (try!) as weight loading is critical and should fail fast if unsuccessful
  static func loadWeights(modelPath: URL) -> [String: MLXArray] {
    // Load raw weights from disk
    let weights = try! MLX.loadArrays(url: modelPath)
    var sanitizedWeights: [String: MLXArray] = [:]

    // Process each weight based on its component prefix
    for (key, value) in weights {
      // Process BERT encoder weights
      if key.hasPrefix("bert") {
        // Skip position_ids as they're not needed for inference
        if key.contains("position_ids") {
          continue
        }
        sanitizedWeights[key] = value
        
      // Process predictor (duration and prosody) weights
      } else if key.hasPrefix("predictor") {
        // F0 projection weights need transposition for proper matrix multiplication
        if key.contains("F0_proj.weight") {
          sanitizedWeights[key] = value.transposed(0, 2, 1)
          
        // N (noise) projection weights need transposition
        } else if key.contains("N_proj.weight") {
          sanitizedWeights[key] = value.transposed(0, 2, 1)
          
        // Weight normalization V parameters need conditional transposition
        } else if key.contains("weight_v") {
          sanitizedWeights[key] = orientWeightV(key: key, value: value, rawWeights: weights)
        } else {
          sanitizedWeights[key] = value
        }

      // Process text encoder weights
      } else if key.hasPrefix("text_encoder") {
        // Weight normalization V parameters need conditional transposition
        if key.contains("weight_v") {
          sanitizedWeights[key] = orientWeightV(key: key, value: value, rawWeights: weights)
        } else {
          sanitizedWeights[key] = value
        }

      // Process decoder weights
      } else if key.hasPrefix("decoder") {
        // Noise convolution weights need transposition
        if key.contains("noise_convs"), key.hasSuffix(".weight") {
          sanitizedWeights[key] = value.transposed(0, 2, 1)

        // Weight normalization V parameters need conditional transposition
        } else if key.contains("weight_v") {
          sanitizedWeights[key] = orientWeightV(key: key, value: value, rawWeights: weights)
        } else {
          sanitizedWeights[key] = value
        }
      }
    }

    return sanitizedWeights
  }

  /// Determines whether a `weight_v` tensor (from PyTorch's weight_norm
  /// decomposition) is already in its correctly-oriented, as-exported
  /// PyTorch shape, or needs its last two axes swapped.
  ///
  /// The previous approach (`checkArrayShape`) guessed based on whether the
  /// kernel dimension looked "square" relative to the channel dimension --
  /// which is wrong whenever a layer's actual kernel size doesn't happen to
  /// coincidentally satisfy that heuristic. Confirmed empirically: both
  /// `decoder.generator.conv_post.weight_v` ([22, 128, 7], a regular Conv1d
  /// with PyTorch's native [out, in, kernel] layout) and
  /// `decoder.generator.ups.0.weight_v` ([512, 256, 20], a ConvTranspose1d
  /// with PyTorch's native [in, out, kernel] layout) are already correctly
  /// oriented as loaded, yet the old heuristic transposed both anyway --
  /// silently swapping the channel and kernel axes and scrambling which
  /// weight tap applies to which input channel. That's not a scale bug,
  /// it's structural corruption of the convolution itself, and it fully
  /// explains generated audio that "doesn't resemble speech at all."
  ///
  /// `weight_g` gives a reliable, non-heuristic answer instead: PyTorch's
  /// weight_norm always stores it with shape [N, 1, 1] where N is the TRUE
  /// channel-axis size in the tensor's native orientation. If `weight_v`'s
  /// own axis 0 already matches that N, it's already correctly oriented.
  private static func orientWeightV(key: String, value: MLXArray, rawWeights: [String: MLXArray]) -> MLXArray {
    let gKey = key.replacingOccurrences(of: "weight_v", with: "weight_g")
    guard let weightG = rawWeights[gKey], value.shape.count == 3, let gAxis0 = weightG.shape.first else {
      // No corresponding weight_g found to cross-check against -- leave
      // as-loaded rather than guess.
      return value
    }

    if value.shape[0] == gAxis0 {
      // Already oriented with the channel axis first, matching weight_g.
      return value
    } else if value.shape[2] == gAxis0 {
      // Channel axis is last; bring it to the front, preserving the
      // relative order of the remaining two axes.
      return value.transposed(2, 0, 1)
    } else {
      // Doesn't match either end -- can't reliably determine orientation.
      // Leave as-loaded rather than apply a blind transpose.
      return value
    }
  }
}
