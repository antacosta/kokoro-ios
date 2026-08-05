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
          sanitizedWeights[key] = checkArrayShape(arr: value) ? value : value.transposed(0, 2, 1)
        } else {
          sanitizedWeights[key] = value
        }

      // Process text encoder weights
      } else if key.hasPrefix("text_encoder") {
        // Weight normalization V parameters need conditional transposition
        if key.contains("weight_v") {
          sanitizedWeights[key] = checkArrayShape(arr: value) ? value : value.transposed(0, 2, 1)
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
          sanitizedWeights[key] = checkArrayShape(arr: value) ? value : value.transposed(0, 2, 1)
        } else {
          sanitizedWeights[key] = value
        }
      }
    }

    return sanitizedWeights
  }

  /// Checks if a 3D weight array has the correct shape and doesn't need transposition.
  ///
  /// REVERTED to this exact heuristic after finding it verbatim in the
  /// authoritative reference: this Swift package is a port of
  /// Blaizzy/mlx-audio's Python MLX implementation of Kokoro (not directly
  /// of the original PyTorch model), and that project's `Decoder.sanitize()`
  /// uses this identical check (`check_array_shape` in
  /// mlx_audio/tts/models/base.py) for every weight_v key -- ups, conv_post,
  /// resblocks, no special-casing. An earlier attempt "fixed" this into an
  /// unconditional axes-1/2 swap based on the wrong assumption that this
  /// heuristic was an arbitrary guess; it isn't -- it's the correct,
  /// intentional logic, and forcing an unconditional swap silently broke
  /// every layer where this legitimately returns true (skip transpose).
  private static func checkArrayShape(arr: MLXArray) -> Bool {
    guard arr.shape.count == 3 else { return false }

    let outChannels = arr.shape[0]
    let kH = arr.shape[1]
    let kW = arr.shape[2]

    return (outChannels >= kH) && (outChannels >= kW) && (kH == kW)
  }
}
