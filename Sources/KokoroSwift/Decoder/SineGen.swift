//
//  Kokoro-tts-lib
//
import Foundation
import MLX
import MLXNN
import MLXRandom

class SineGen {
  private let sineAmp: Float
  private let noiseStd: Float
  private let harmonicNum: Int
  private let dim: Int
  private let samplingRate: Int
  private let voicedThreshold: Float
  private let upsampleScale: Float

  init(
    sampRate: Int,
    upsampleScale: Float,
    harmonicNum: Int = 0,
    sineAmp: Float = 0.1,
    noiseStd: Float = 0.003,
    voicedThreshold: Float = 0
  ) {
    self.sineAmp = sineAmp
    self.noiseStd = noiseStd
    self.harmonicNum = harmonicNum
    dim = harmonicNum + 1
    samplingRate = sampRate
    self.voicedThreshold = voicedThreshold
    self.upsampleScale = upsampleScale
  }

  private func _f02uv(_ f0: MLXArray) -> MLXArray {
    let arr = f0 .> voicedThreshold
    return arr.asType(.float32)
  }

  private func _f02sine(_ f0Values: MLXArray) -> MLXArray {
    // Ignore integer part (% 1 is there for a purpose :)
    var radValues = (f0Values / Float(samplingRate)) % 1

    // Random phase noise
    let randIni = MLXRandom.normal([f0Values.shape[0], f0Values.shape[2]])
    randIni[0..., 0] = MLXArray(0.0)
    radValues[0 ..< radValues.shape[0], 0, 0 ..< radValues.shape[2]] = radValues[0 ..< radValues.shape[0], 0, 0 ..< radValues.shape[2]] + randIni

    radValues = interpolate(
      input: radValues.transposed(0, 2, 1),
      scaleFactor: [1 / Float(upsampleScale)],
      mode: "linear"
    ).transposed(0, 2, 1)
    let radFlat = radValues.asArray(Float.self)
    print("[KokoroDiag] SineGen radValues(downsampled) count=\(radFlat.count) min=\(radFlat.min() ?? 0) max=\(radFlat.max() ?? 0) first5=\(radFlat.prefix(5)) last5=\(radFlat.suffix(5))")

    var phase = MLX.cumsum(radValues, axis: 1) * 2 * Float.pi
    let phaseBeforeFlat = phase.asArray(Float.self)
    print("[KokoroDiag] SineGen phase(before upsample) min=\(phaseBeforeFlat.min() ?? 0) max=\(phaseBeforeFlat.max() ?? 0) first5=\(phaseBeforeFlat.prefix(5))")

    phase = interpolate(
      input: phase.transposed(0, 2, 1) * Float(upsampleScale),
      scaleFactor: [Float(upsampleScale)],
      mode: "linear"
    ).transposed(0, 2, 1)
    let phaseAfterFlat = phase.asArray(Float.self)
    print("[KokoroDiag] SineGen phase(after upsample) count=\(phaseAfterFlat.count) min=\(phaseAfterFlat.min() ?? 0) max=\(phaseAfterFlat.max() ?? 0) first10=\(phaseAfterFlat.prefix(10)) last10=\(phaseAfterFlat.suffix(10))")

    return MLX.sin(phase)
  }

  func callAsFunction(_ f0: MLXArray) -> (MLXArray, MLXArray, MLXArray) {
    // Fundamental component
    let range = MLXArray(1 ... harmonicNum + 1).asType(.float32)
    let fn = f0 * range.reshaped([1, 1, range.shape[0]])

    // Generate sine waveforms
    var sineWaves = _f02sine(fn) * sineAmp

    // Generate UV signal
    var uv = _f02uv(f0)

    // Correction: PR #814 (which the previous version of this comment
    // cited as "upstream-confirmed") was actually REJECTED and CLOSED by
    // the mlx-audio maintainer -- the -10/10 exp clamp was called "pretty
    // arbitrary", and this trim-both-to-min-length approach was never
    // merged either. The actual merged fix (commit cc30ce2, "Fix SineGen
    // length alignment") only ever touches sineWaves, matching it to f0's
    // length -- trimming if longer, zero-padding if shorter -- and never
    // shortens uv/f0. Trimming uv (as the rejected PR did) would silently
    // discard real, correctly-computed voiced/unvoiced samples whenever
    // sineWaves came out shorter, which is a regression, not a fix.
    let targetLen = f0.shape[1]
    if sineWaves.shape[1] > targetLen {
      sineWaves = sineWaves[0..., 0 ..< targetLen, 0...]
    } else if sineWaves.shape[1] < targetLen {
      sineWaves = MLX.padded(sineWaves, widths: [IntOrPair([0, 0]), IntOrPair([0, targetLen - sineWaves.shape[1]]), IntOrPair([0, 0])])
    }

    // Generate noise
    let noiseAmp = uv * noiseStd + (1 - uv) * sineAmp / 3
    let noise = noiseAmp * MLXRandom.normal(sineWaves.shape)

    let result = sineWaves * uv + noise
    return (result, uv, noise)
  }
}
