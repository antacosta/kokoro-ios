//
//  Kokoro-tts-lib
//
import Foundation
import MLX
import MLXNN

class Generator {
  let numKernels: Int
  let numUpsamples: Int
  let mSource: SourceModuleHnNSF
  let f0Upsample: Upsample
  let postNFFt: Int
  var noiseConvs: [Conv1dInference]
  var noiseRes: [AdaINResBlock1]
  var ups: [ConvWeighted]
  var resBlocks: [AdaINResBlock1]
  let convPost: ConvWeighted
  let reflectionPad: ReflectionPad1d
  let stft: MLXSTFT

  init(weights: [String: MLXArray],
       styleDim: Int,
       resblockKernelSizes: [Int],
       upsampleRates: [Int],
       upsampleInitialChannel: Int,
       resblockDilationSizes: [[Int]],
       upsampleKernelSizes: [Int],
       genIstftNFft: Int,
       genIstftHopSize: Int)
  {
    numKernels = resblockKernelSizes.count
    numUpsamples = upsampleRates.count

    let upsampleScaleNum = MLX.product(MLXArray(upsampleRates)) * genIstftHopSize
    let upsampleScaleNumVal: Int = upsampleScaleNum.item()

    mSource = SourceModuleHnNSF(
      weights: weights,
      samplingRate: KokoroTTS.Constants.samplingRate,
      upsampleScale: upsampleScaleNum.item(),
      harmonicNum: 8,
      voicedThreshold: 10
    )

    f0Upsample = Upsample(scaleFactor: .float(Float(upsampleScaleNumVal)))

    noiseConvs = []
    noiseRes = []
    ups = []

    for (i, (u, k)) in zip(upsampleRates, upsampleKernelSizes).enumerated() {
      let upConv = ConvWeighted(
        weightG: weights["decoder.generator.ups.\(i).weight_g"]!,
        weightV: weights["decoder.generator.ups.\(i).weight_v"]!,
        bias: weights["decoder.generator.ups.\(i).bias"]!,
        stride: u,
        padding: (k - u) / 2
      )
      upConv.debugLabel = "ups.\(i)"
      ups.append(upConv)
    }

    resBlocks = []
    for i in 0 ..< ups.count {
      let ch = upsampleInitialChannel / Int(pow(2.0, Double(i + 1)))
      for (j, (k, d)) in zip(resblockKernelSizes, resblockDilationSizes).enumerated() {
        resBlocks.append(
          AdaINResBlock1(
            weights: weights,
            weightPrefixKey: "decoder.generator.resblocks.\((i * resblockKernelSizes.count) + j)",
            channels: ch,
            kernelSize: k,
            dilation: d,
            styleDim: styleDim
          )
        )
      }

      let cCur = ch
      if i + 1 < upsampleRates.count {
        let strideF0: Int = MLX.product(MLXArray(upsampleRates)[(i + 1)...]).item()
        noiseConvs.append(
          Conv1dInference(
            inputChannels: genIstftNFft + 2,
            outputChannels: cCur,
            kernelSize: strideF0 * 2,
            stride: strideF0,
            padding: (strideF0 + 1) / 2,
            weight: weights["decoder.generator.noise_convs.\(i).weight"]!,
            bias: weights["decoder.generator.noise_convs.\(i).bias"]!
          )
        )

        noiseRes.append(
          AdaINResBlock1(
            weights: weights,
            weightPrefixKey: "decoder.generator.noise_res.\(i)",
            channels: cCur,
            kernelSize: 7,
            dilation: [1, 3, 5],
            styleDim: styleDim
          )
        )
      } else {
        noiseConvs.append(
          Conv1dInference(
            inputChannels: genIstftNFft + 2,
            outputChannels: cCur,
            kernelSize: 1,
            weight: weights["decoder.generator.noise_convs.\(i).weight"]!,
            bias: weights["decoder.generator.noise_convs.\(i).bias"]!
          )
        )
        noiseRes.append(
          AdaINResBlock1(
            weights: weights,
            weightPrefixKey: "decoder.generator.noise_res.\(i)",
            channels: cCur,
            kernelSize: 11,
            dilation: [1, 3, 5],
            styleDim: styleDim
          )
        )
      }
    }

    postNFFt = genIstftNFft

    convPost = ConvWeighted(
      weightG: weights["decoder.generator.conv_post.weight_g"]!,
      weightV: weights["decoder.generator.conv_post.weight_v"]!,
      bias: weights["decoder.generator.conv_post.bias"]!,
      stride: 1,
      padding: 3
    )
    convPost.debugLabel = "conv_post"

    reflectionPad = ReflectionPad1d(padding: (1, 0))

    stft = MLXSTFT(
      filterLength: genIstftNFft,
      hopLength: genIstftHopSize,
      winLength: genIstftNFft
    )
  }

  func callAsFunction(_ x: MLXArray, _ s: MLXArray, _ F0Curve: MLXArray) -> MLXArray {
    print("[KokoroDiag] generator x=\(x.shape) F0Curve=\(F0Curve.shape)")
    var f0New = F0Curve[.newAxis, 0..., 0...].transposed(0, 2, 1)
    f0New = f0Upsample(f0New)
    print("[KokoroDiag] generator f0New(upsampled)=\(f0New.shape)")

    var (harSource, _, _) = mSource(f0New)
    let harSourceFlat = harSource.asArray(Float.self)
    let harNonFinite = harSourceFlat.filter { !$0.isFinite }.count
    print("[KokoroDiag] generator harSource=\(harSource.shape) nonFiniteCount=\(harNonFinite) min=\(harSourceFlat.min() ?? 0) max=\(harSourceFlat.max() ?? 0)")

    harSource = MLX.squeezed(harSource.transposed(0, 2, 1), axis: 1)
    let (harSpec, harPhase) = stft.transform(inputData: harSource)
    print("[KokoroDiag] generator harSpec=\(harSpec.shape) harPhase=\(harPhase.shape)")

    var har = MLX.concatenated([harSpec, harPhase], axis: 1)
    har = MLX.swappedAxes(har, 2, 1)

    var newX = x
    for i in 0 ..< numUpsamples {
      newX = LeakyReLU(negativeSlope: 0.1)(newX)
      var xSource = noiseConvs[i](har)
      xSource = MLX.swappedAxes(xSource, 2, 1)
      xSource = noiseRes[i](xSource, s)

      newX = MLX.swappedAxes(newX, 2, 1)
      newX = ups[i](newX, conv: MLX.convTransposed1d)
      newX = MLX.swappedAxes(newX, 2, 1)
      print("[KokoroDiag] generator upsample[\(i)] newX=\(newX.shape) xSource=\(xSource.shape)")

      if i == numUpsamples - 1 {
        newX = reflectionPad(newX)
      }
      newX = newX + xSource

      var xs: MLXArray?
      for j in 0 ..< numKernels {
        if xs == nil {
          xs = resBlocks[i * numKernels + j](newX, s)
        } else {
          let temp = resBlocks[i * numKernels + j](newX, s)
          xs = xs! + temp
        }
      }
      newX = xs! / numKernels
    }

    newX = LeakyReLU(negativeSlope: 0.01)(newX)

    newX = MLX.swappedAxes(newX, 2, 1)
    newX = convPost(newX, conv: MLX.conv1d)
    newX = MLX.swappedAxes(newX, 2, 1)
    print("[KokoroDiag] generator postConv newX=\(newX.shape)")

    // Upstream-confirmed bug (Blaizzy/mlx-audio #815, fixed in PR #814):
    // convPost's raw output can reach magnitudes around ~1e11 for some
    // inputs. Exponentiating that directly overflows float32 to inf,
    // which then propagates as NaN through the iSTFT reconstruction --
    // and even short of literal overflow, an unclamped, occasionally
    // very large log-magnitude drives the reconstructed waveform's
    // amplitude far outside any normal audio range, producing harsh,
    // distorted output. Clamping the log-magnitude to [-10, 10] before
    // exp() (matching common TTS spectrogram processing practice, and
    // the exact bound used in the upstream fix) keeps this bounded.
    let logMagnitude = MLX.clip(newX[0..., 0 ..< (postNFFt / 2 + 1), 0...], min: -10.0, max: 10.0)
    let spec = MLX.exp(logMagnitude)
    let phase = MLX.sin(newX[0..., (postNFFt / 2 + 1)..., 0...])
    let specFlat = spec.asArray(Float.self)
    print("[KokoroDiag] generator spec nonFiniteCount=\(specFlat.filter { !$0.isFinite }.count) min=\(specFlat.min() ?? 0) max=\(specFlat.max() ?? 0)")

    let result = stft.inverse(magnitude: spec, phase: phase)
    let resultFlat = result.asArray(Float.self)
    print("[KokoroDiag] generator result=\(result.shape) nonFiniteCount=\(resultFlat.filter { !$0.isFinite }.count) min=\(resultFlat.min() ?? 0) max=\(resultFlat.max() ?? 0)")

    // Is the extreme range confined to the edges (an ISTFT overlap-add
    // edge/trim issue) or does it show up throughout (a more pervasive
    // per-frame-boundary issue)? Sample a handful of windows across the
    // signal to compare.
    let n = resultFlat.count
    let edgeTrim = min(200, n / 4)
    if n > edgeTrim * 2 {
      let middle = resultFlat[edgeTrim ..< (n - edgeTrim)]
      print("[KokoroDiag] generator result excluding \(edgeTrim)-sample edges: min=\(middle.min() ?? 0) max=\(middle.max() ?? 0)")
    }
    let windowSize = max(1, n / 10)
    for w in 0 ..< 10 {
      let start = w * windowSize
      let end = min(n, start + windowSize)
      guard start < end else { continue }
      let slice = resultFlat[start ..< end]
      print("[KokoroDiag] generator result window[\(w)] range=[\(start),\(end)) min=\(slice.min() ?? 0) max=\(slice.max() ?? 0)")
    }
    return result
  }
}
