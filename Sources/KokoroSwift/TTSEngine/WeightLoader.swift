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
          sanitizedWeights[key] = orientWeightV(value)
        } else {
          sanitizedWeights[key] = value
        }

      // Process text encoder weights
      } else if key.hasPrefix("text_encoder") {
        // Weight normalization V parameters need conditional transposition
        if key.contains("weight_v") {
          sanitizedWeights[key] = orientWeightV(value)
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
          sanitizedWeights[key] = orientWeightV(value)
        } else {
          sanitizedWeights[key] = value
        }
      }
    }

    return sanitizedWeights
  }

  /// Converts a `weight_v` tensor (from PyTorch's weight_norm
  /// decomposition) from PyTorch's native per-layer-type shape into the
  /// shape MLX's conv/convTransposed ops expect.
  ///
  /// The previous approach (`checkArrayShape`) guessed per-layer whether to
  /// swap axes based on whether the kernel dimension looked "square"
  /// relative to the channel dimension -- inconsistent, and wrong whenever
  /// a layer's actual kernel size doesn't coincidentally satisfy that
  /// check. Confirmed against the reference PyTorch model (hexgrad/kokoro):
  /// every weight_v tensor here uses weight_norm's default dim=0, so its
  /// channel axis is always already at position 0 in PyTorch's native
  /// layout -- [out, in, kernel] for Conv1d, [in, out, kernel] for
  /// ConvTranspose1d, confirmed via decoder.generator.conv_post.weight_v
  /// ([22, 128, 7]), decoder.generator.ups.0.weight_v ([512, 256, 20]),
  /// and predictor.F0.0.conv1.weight_v ([512, 512, 3]) all matching their
  /// reference nn.Conv1d/nn.ConvTranspose1d definitions exactly. MLX's
  /// convention keeps that same channel axis first but wants kernel
  /// second: [channel, kernel, other]. So every weight_v needs the same,
  /// unconditional axes-1/2 swap -- never zero, and never a full reverse
  /// (which is what the old heuristic's wrong branch, or ConvWeighted's
  /// own runtime `x.shape.last == weight.shape.last` fallback comparing
  /// channel-count to kernel-size, would otherwise apply on top and
  /// scramble it a second time).
  private static func orientWeightV(_ value: MLXArray) -> MLXArray {
    guard value.shape.count == 3 else { return value }
    return value.transposed(0, 2, 1)
  }
}
