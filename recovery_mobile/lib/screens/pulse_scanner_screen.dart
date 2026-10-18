import 'dart:async';
import 'dart:math' as math;
import 'package:camera/camera.dart';
import 'package:flutter/material.dart';
import '../models/recovery_result.dart';
import '../services/fatigue_api_service.dart';
import '../widgets/pulse_waveform_painter.dart';

enum ScannerState {
  idle,
  fingerDetecting,
  measuring,
  analyzing,
  resultsView,
}

class PulseScannerScreen extends StatefulWidget {
  final List<CameraDescription> cameras;

  const PulseScannerScreen({super.key, required this.cameras});

  @override
  State<PulseScannerScreen> createState() => _PulseScannerScreenState();
}

class _PulseScannerScreenState extends State<PulseScannerScreen>
    with SingleTickerProviderStateMixin {
  // State Machine
  ScannerState _state = ScannerState.idle;

  // Hardware Camera Controller
  CameraController? _cameraController;

  // Optical Signal Buffers
  final List<double> _rawSignals = [];
  final List<double> _timestampsMs = [];
  final List<double> _displayWaveformBuffer = [];

  // Measurement Timing
  static const int _targetDurationSec = 45;
  int _secondsRemaining = _targetDurationSec;
  Timer? _countdownTimer;
  int _lastFrameTimestampMs = 0;

  // Fingertip Placement Validation & Pause Gating
  bool _isFingerCoveringLens = false;
  bool _isPausedDueToFingerLoss = false;
  int _consecutiveValidFrames = 0;
  int _contactLostFrames = 0;
  double _accumulatedValidMs = 0.0;
  String _guidanceMessage = 'Position index finger gently over rear camera & flash';

  // Animation & Visuals
  late AnimationController _animController;
  final FatigueApiService _apiService = FatigueApiService();
  RecoveryResult? _analysisResult;
  String? _errorMessage;

  // Simulation mode for emulators/desktops without camera hardware
  bool _isSimulationMode = false;
  Timer? _simulationTimer;

  @override
  void initState() {
    super.initState();
    _animController = AnimationController(
      vsync: this,
      duration: const Duration(milliseconds: 1500),
    )..repeat();

    _initializeCamera();
  }

  @override
  void dispose() {
    _countdownTimer?.cancel();
    _simulationTimer?.cancel();
    _animController.dispose();
    _disposeCamera();
    super.dispose();
  }

  Future<void> _disposeCamera() async {
    if (_cameraController != null) {
      if (_cameraController!.value.isStreamingImages) {
        await _cameraController!.stopImageStream();
      }
      await _cameraController!.dispose();
      _cameraController = null;
    }
  }

  Future<void> _initializeCamera() async {
    if (widget.cameras.isEmpty) {
      setState(() {
        _isSimulationMode = true;
        _guidanceMessage = 'No camera hardware found. Simulation Mode activated.';
      });
      return;
    }

    try {
      // Find the primary rear camera
      final rearCamera = widget.cameras.firstWhere(
        (cam) => cam.lensDirection == CameraLensDirection.back,
        orElse: () => widget.cameras.first,
      );

      final controller = CameraController(
        rearCamera,
        ResolutionPreset.low, // Low resolution minimizes CPU & memory overhead
        enableAudio: false,
        imageFormatGroup: ImageFormatGroup.yuv420,
      );

      await controller.initialize();

      if (!mounted) return;
      setState(() {
        _cameraController = controller;
      });
    } catch (e) {
      debugPrint('Camera init error: $e');
      setState(() {
        _isSimulationMode = true;
        _guidanceMessage = 'Camera init failed. Switching to physiological simulation.';
      });
    }
  }

  // --- Scan Lifecycle Actions ---

  Future<void> _startMeasurementFlow() async {
    setState(() {
      _errorMessage = null;
      _analysisResult = null;
      _rawSignals.clear();
      _timestampsMs.clear();
      _displayWaveformBuffer.clear();
      _secondsRemaining = _targetDurationSec;
      _isFingerCoveringLens = false;
      _isPausedDueToFingerLoss = false;
      _consecutiveValidFrames = 0;
      _contactLostFrames = 0;
      _accumulatedValidMs = 0.0;
      _state = ScannerState.fingerDetecting;
      _guidanceMessage = 'Locking camera sensors & detecting fingertip pulse...';
    });

    if (_isSimulationMode || _cameraController == null) {
      _startSimulationScan();
      return;
    }

    try {
      // 1. Engage Torch Flashlight
      await _cameraController!.setFlashMode(FlashMode.torch);

      // 2. Lock Auto-Exposure to prevent AGC gain oscillation
      try {
        await _cameraController!.setExposureMode(ExposureMode.locked);
      } catch (e) {
        debugPrint('Could not lock exposure: $e');
      }

      // 3. Lock Auto-Focus to avoid mechanical hunting
      try {
        await _cameraController!.setFocusMode(FocusMode.locked);
      } catch (e) {
        debugPrint('Could not lock focus: $e');
      }

      // 4. Begin Image Stream
      _lastFrameTimestampMs = DateTime.now().millisecondsSinceEpoch;
      await _cameraController!.startImageStream(_processCameraFrame);
    } catch (e) {
      setState(() {
        _errorMessage = 'Hardware camera lock error: $e';
        _state = ScannerState.idle;
      });
    }
  }

  /// Low-overhead optical frame processor throttled to ~30 FPS with real-time pause gating
  void _processCameraFrame(CameraImage image) {
    final nowMs = DateTime.now().millisecondsSinceEpoch;
    // Throttle to max 30 FPS (~33ms period) to maintain low CPU footprint
    if (nowMs - _lastFrameTimestampMs < 30) {
      return;
    }
    final frameDtMs = _lastFrameTimestampMs > 0 ? (nowMs - _lastFrameTimestampMs) : 33;
    _lastFrameTimestampMs = nowMs;

    // Subsample frame center to extract luminance & red saturation
    final metrics = _extractFrameOpticalMetrics(image);
    final double luminance = metrics.luminance;
    final bool isFingerPresent = metrics.isFingerCovering;

    if (!mounted) return;

    if (_state == ScannerState.fingerDetecting) {
      if (isFingerPresent) {
        _consecutiveValidFrames++;
        if (_consecutiveValidFrames >= 20) {
          // ~0.7s of stable, validated fingertip transillumination
          setState(() {
            _state = ScannerState.measuring;
            _isFingerCoveringLens = true;
            _isPausedDueToFingerLoss = false;
            _contactLostFrames = 0;
            _accumulatedValidMs = 0.0;
            _secondsRemaining = _targetDurationSec;
            _guidanceMessage = 'Fingertip locked. Keep still for 45s autonomic scan...';
          });
        } else {
          setState(() {
            _isFingerCoveringLens = true;
            _guidanceMessage = 'Detecting pulse: hold finger steady... (${(_consecutiveValidFrames / 20 * 100).toInt()}%)';
          });
        }
      } else {
        _consecutiveValidFrames = 0;
        setState(() {
          _isFingerCoveringLens = false;
          _guidanceMessage = 'Place index fingertip firmly covering both camera & flash';
        });
      }
    } else if (_state == ScannerState.measuring) {
      if (isFingerPresent) {
        _contactLostFrames = 0;
        final clampedDt = frameDtMs.clamp(15, 55);
        _accumulatedValidMs += clampedDt;

        final newSecondsRemaining = (_targetDurationSec - (_accumulatedValidMs / 1000.0)).ceil().clamp(0, _targetDurationSec);

        setState(() {
          _isFingerCoveringLens = true;
          _isPausedDueToFingerLoss = false;
          _secondsRemaining = newSecondsRemaining;
          _guidanceMessage = 'Capturing microvascular pulse wave... ($_secondsRemaining s)';

          // Record valid, verified optical pulse sample
          _timestampsMs.add(_accumulatedValidMs);
          _rawSignals.add(luminance);

          // Update rolling waveform visualizer
          _displayWaveformBuffer.add(luminance);
          if (_displayWaveformBuffer.length > 80) {
            _displayWaveformBuffer.removeAt(0);
          }
        });

        if (_accumulatedValidMs >= (_targetDurationSec * 1000.0)) {
          _concludeMeasurement();
        }
      } else {
        // Finger contact lost! PAUSE IMMEDIATELY.
        _contactLostFrames++;

        // If contact lost for > ~8 seconds (240 frames at 30 FPS)
        if (_contactLostFrames > 240) {
          _resetToIdle();
          setState(() {
            _errorMessage = 'Scan cancelled: Fingertip was removed for over 8 seconds. Please keep your finger firmly covering the camera and flashlight throughout the test.';
          });
          return;
        }

        setState(() {
          _isFingerCoveringLens = false;
          _isPausedDueToFingerLoss = true;
          _guidanceMessage = 'FINGERTIP CONTACT LOST! PAUSED - Press finger firmly on lens & flash to resume.';
        });
      }
    }
  }

  /// Extracts mean luminance and verifies red-channel dominance
  _OpticalMetrics _extractFrameOpticalMetrics(CameraImage image) {
    double totalY = 0.0;
    int sampleCount = 0;

    if (image.format.group == ImageFormatGroup.yuv420 && image.planes.length >= 3) {
      final yPlane = image.planes[0];
      final uPlane = image.planes[1]; // U is Cb (Blue chrominance difference)
      final vPlane = image.planes[2]; // V is Cr (Red chrominance difference)
      final bytesY = yPlane.bytes;
      final bytesU = uPlane.bytes;
      final bytesV = vPlane.bytes;

      // Subsample every 16th byte to minimize overhead
      const stepY = 16;
      for (int i = 0; i < bytesY.length; i += stepY) {
        totalY += bytesY[i];
        sampleCount++;
      }

      int totalU = 0;
      int uCount = 0;
      const stepChroma = 16;
      for (int i = 0; i < bytesU.length; i += stepChroma) {
        totalU += bytesU[i];
        uCount++;
      }

      int totalV = 0;
      int vCount = 0;
      for (int i = 0; i < bytesV.length; i += stepChroma) {
        totalV += bytesV[i];
        vCount++;
      }

      final meanY = sampleCount > 0 ? totalY / sampleCount : 0.0;
      final meanU = uCount > 0 ? totalU / uCount : 128.0;
      final meanV = vCount > 0 ? totalV / vCount : 128.0;
      final chromaDiff = meanV - meanU;

      // Strict tissue transillumination check:
      // 1. Red chrominance (V/Cr) must be strongly elevated (>= 158.0)
      // 2. Blue chrominance (U/Cb) must be strongly absorbed (<= 122.0)
      // 3. Difference (V - U) must exceed 40.0 (confirms biological red transillumination)
      // 4. Mean luminance Y must be within transilluminated tissue range [35.0, 245.0]
      final bool isFinger = (meanV >= 158.0) &&
          (meanU <= 122.0) &&
          (chromaDiff >= 40.0) &&
          (meanY >= 35.0 && meanY <= 245.0);

      return _OpticalMetrics(
        luminance: meanY,
        isFingerCovering: isFinger,
        redChroma: meanV,
        blueChroma: meanU,
        chromaDiff: chromaDiff,
      );
    } else {
      // Fallback for BGRA8888 (e.g. iOS) or other formats
      final plane = image.planes[0];
      final bytes = plane.bytes;
      double totalR = 0.0;
      double totalG = 0.0;
      double totalB = 0.0;

      const step = 24; // Sample every 6th pixel (4 bytes/pixel)
      for (int i = 0; i < bytes.length - 3; i += step) {
        final b = bytes[i];
        final g = bytes[i + 1];
        final r = bytes[i + 2];
        totalR += r;
        totalG += g;
        totalB += b;
        sampleCount++;
      }

      final meanR = sampleCount > 0 ? totalR / sampleCount : 0.0;
      final meanG = sampleCount > 0 ? totalG / sampleCount : 0.0;
      final meanB = sampleCount > 0 ? totalB / sampleCount : 0.0;

      // Hemoglobin absorbs green strongly; red transmits brightly through skin
      final luminance = (0.299 * meanR) + (0.587 * meanG) + (0.114 * meanB);
      final isFinger = (meanR >= 135.0) &&
          (meanR >= meanG * 1.5) &&
          (meanR >= meanB * 2.2) &&
          (meanB <= 95.0);

      return _OpticalMetrics(
        luminance: meanG > 0 ? meanG : luminance,
        isFingerCovering: isFinger,
        redChroma: meanR,
        blueChroma: meanB,
        chromaDiff: meanR - meanB,
      );
    }
  }

  void _startCountdownTimer() {
    _countdownTimer?.cancel();
    _countdownTimer = Timer.periodic(const Duration(seconds: 1), (timer) {
      if (!mounted) return;
      setState(() {
        if (_secondsRemaining > 1) {
          _secondsRemaining--;
        } else {
          _secondsRemaining = 0;
          timer.cancel();
          _concludeMeasurement();
        }
      });
    });
  }

  Future<void> _concludeMeasurement() async {
    _countdownTimer?.cancel();
    _simulationTimer?.cancel();

    // Stop camera streaming and turn off flashlight
    if (_cameraController != null && _cameraController!.value.isStreamingImages) {
      await _cameraController!.stopImageStream();
      await _cameraController!.setFlashMode(FlashMode.off);
    }

    // Verify minimum valid data collection
    if (_rawSignals.length < 600 || _timestampsMs.length < 600) {
      setState(() {
        _errorMessage = 'Scan incomplete: Less than 20 seconds of stable pulse data was captured. Please hold your finger firmly over both camera lens and flashlight for the entire duration.';
        _state = ScannerState.idle;
      });
      return;
    }

    setState(() {
      _state = ScannerState.analyzing;
      _guidanceMessage = 'Transmitting raw optical stream to SciPy DSP service...';
    });

    try {
      final result = await _apiService.analyzeFatigue(
        timestampsMs: _timestampsMs,
        rawSignals: _rawSignals,
      );

      setState(() {
        _analysisResult = result;
        _state = ScannerState.resultsView;
      });
    } on ApiException catch (e) {
      setState(() {
        _errorMessage = e.message;
        _state = ScannerState.idle;
      });
    } catch (e) {
      setState(() {
        _errorMessage = 'Analysis failed: $e';
        _state = ScannerState.idle;
      });
    }
  }

  /// Synthetic physiological simulation when camera hardware is unavailable
  void _startSimulationScan() {
    _consecutiveValidFrames = 0;
    Timer(const Duration(milliseconds: 800), () {
      if (!mounted) return;
      setState(() {
        _state = ScannerState.measuring;
        _isFingerCoveringLens = true;
        _isPausedDueToFingerLoss = false;
        _accumulatedValidMs = 0.0;
        _guidanceMessage = 'Simulating 45s physiological pulse stream...';
      });
      _startCountdownTimer();

      // Emit simulated pulses at 30 Hz
      int tick = 0;
      _simulationTimer = Timer.periodic(const Duration(milliseconds: 33), (timer) {
        if (!mounted || _state != ScannerState.measuring) {
          timer.cancel();
          return;
        }
        tick++;
        final tSec = tick * 0.0333;
        final elapsedMs = tSec * 1000.0;

        // Cardiac wave with natural HRV & respiratory modulation
        final cardiac = 0.8 * math.sin(2 * math.pi * 1.15 * tSec) +
            0.35 * math.sin(4 * math.pi * 1.15 * tSec + 0.5) +
            0.15 * math.sin(6 * math.pi * 1.15 * tSec + 1.2);
        final rsa = 0.2 * math.sin(2 * math.pi * 0.25 * tSec);
        final noise = (math.Random().nextDouble() - 0.5) * 0.08;
        final signalVal = 120.0 + cardiac + rsa + noise;

        setState(() {
          _timestampsMs.add(elapsedMs);
          _rawSignals.add(signalVal);
          _displayWaveformBuffer.add(signalVal);
          if (_displayWaveformBuffer.length > 80) {
            _displayWaveformBuffer.removeAt(0);
          }
        });
      });
    });
  }

  void _resetToIdle() {
    _countdownTimer?.cancel();
    _simulationTimer?.cancel();
    setState(() {
      _state = ScannerState.idle;
      _errorMessage = null;
      _analysisResult = null;
      _secondsRemaining = _targetDurationSec;
      _displayWaveformBuffer.clear();
      _rawSignals.clear();
      _timestampsMs.clear();
      _isFingerCoveringLens = false;
      _isPausedDueToFingerLoss = false;
      _consecutiveValidFrames = 0;
      _contactLostFrames = 0;
      _accumulatedValidMs = 0.0;
      _guidanceMessage = 'Position index finger gently over rear camera & flash';
    });
  }

  // --- UI Components ---

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      backgroundColor: const Color(0xFF0A0E17),
      appBar: AppBar(
        backgroundColor: const Color(0xFF101726),
        elevation: 0,
        title: Row(
          children: [
            Container(
              padding: const EdgeInsets.all(6),
              decoration: BoxDecoration(
                color: const Color(0xFF00E676).withValues(alpha: 0.15),
                borderRadius: BorderRadius.circular(8),
              ),
              child: const Icon(Icons.monitor_heart_rounded, color: Color(0xFF00E676), size: 20),
            ),
            const SizedBox(width: 10),
            const Text(
              'RECOVERY PULSE',
              style: TextStyle(
                fontSize: 16,
                fontWeight: FontWeight.w800,
                letterSpacing: 1.2,
                color: Colors.white,
              ),
            ),
          ],
        ),
        actions: [
          IconButton(
            tooltip: 'Configure Backend URL',
            icon: const Icon(Icons.settings_ethernet_rounded, color: Color(0xFF94A3B8)),
            onPressed: _showSettingsDialog,
          ),
        ],
      ),
      body: SafeArea(
        child: SingleChildScrollView(
          padding: const EdgeInsets.all(20.0),
          child: Column(
            crossAxisAlignment: CrossAxisAlignment.stretch,
            children: [
              if (_errorMessage != null) _buildErrorBanner(),
              if (_state == ScannerState.resultsView && _analysisResult != null)
                _buildResultsView()
              else
                _buildScannerActiveFlow(),
            ],
          ),
        ),
      ),
    );
  }

  Widget _buildScannerActiveFlow() {
    final progress = (_targetDurationSec - _secondsRemaining) / _targetDurationSec;

    return Column(
      children: [
        // Guidance Pill with Instant Contact Status
        AnimatedContainer(
          duration: const Duration(milliseconds: 300),
          padding: const EdgeInsets.symmetric(horizontal: 16, vertical: 10),
          decoration: BoxDecoration(
            color: _isPausedDueToFingerLoss
                ? const Color(0xFFFFB300).withValues(alpha: 0.15)
                : (_isFingerCoveringLens
                    ? const Color(0xFF00E676).withValues(alpha: 0.12)
                    : const Color(0xFF1E293B)),
            borderRadius: BorderRadius.circular(30),
            border: Border.all(
              color: _isPausedDueToFingerLoss
                  ? const Color(0xFFFFB300).withValues(alpha: 0.7)
                  : (_isFingerCoveringLens
                      ? const Color(0xFF00E676).withValues(alpha: 0.5)
                      : const Color(0xFF334155)),
            ),
          ),
          child: Row(
            mainAxisSize: MainAxisSize.min,
            children: [
              Icon(
                _isPausedDueToFingerLoss
                    ? Icons.warning_amber_rounded
                    : (_isFingerCoveringLens ? Icons.fingerprint_rounded : Icons.info_outline_rounded),
                size: 18,
                color: _isPausedDueToFingerLoss
                    ? const Color(0xFFFFB300)
                    : (_isFingerCoveringLens ? const Color(0xFF00E676) : const Color(0xFF94A3B8)),
              ),
              const SizedBox(width: 8),
              Flexible(
                child: Text(
                  _guidanceMessage,
                  style: TextStyle(
                    fontSize: 13,
                    fontWeight: FontWeight.w600,
                    color: _isPausedDueToFingerLoss
                        ? const Color(0xFFFFB300)
                        : (_isFingerCoveringLens ? const Color(0xFF00E676) : const Color(0xFFCBD5E1)),
                  ),
                ),
              ),
            ],
          ),
        ),
        const SizedBox(height: 28),

        // 45-Second Circular Progress Ring & Countdown
        Stack(
          alignment: Alignment.center,
          children: [
            SizedBox(
              width: 220,
              height: 220,
              child: CircularProgressIndicator(
                value: _state == ScannerState.measuring ? progress : (_state == ScannerState.analyzing ? null : 0.0),
                strokeWidth: 8,
                backgroundColor: const Color(0xFF1A2234),
                valueColor: AlwaysStoppedAnimation<Color>(
                  _isPausedDueToFingerLoss
                      ? const Color(0xFFFFB300)
                      : const Color(0xFF00F0FF),
                ),
              ),
            ),
            Container(
              width: 190,
              height: 190,
              decoration: BoxDecoration(
                shape: BoxShape.circle,
                gradient: const RadialGradient(
                  colors: [
                    Color(0xFF1E293B),
                    Color(0xFF0F172A),
                  ],
                ),
                boxShadow: [
                  BoxShadow(
                    color: (_state == ScannerState.measuring
                            ? (_isPausedDueToFingerLoss ? const Color(0xFFFFB300) : const Color(0xFF00F0FF))
                            : Colors.black)
                        .withValues(alpha: 0.15),
                    blurRadius: 20,
                    spreadRadius: 2,
                  ),
                ],
              ),
              child: Column(
                mainAxisAlignment: MainAxisAlignment.center,
                children: [
                  if (_state == ScannerState.analyzing) ...[
                    const CircularProgressIndicator(
                      valueColor: AlwaysStoppedAnimation<Color>(Color(0xFF00E676)),
                    ),
                    const SizedBox(height: 12),
                    const Text(
                      'COMPUTING DSP',
                      style: TextStyle(
                        fontSize: 12,
                        fontWeight: FontWeight.w800,
                        color: Color(0xFF00E676),
                        letterSpacing: 1.1,
                      ),
                    ),
                  ] else if (_isPausedDueToFingerLoss) ...[
                    const Icon(
                      Icons.pause_circle_filled_rounded,
                      color: Color(0xFFFFB300),
                      size: 46,
                    ),
                    const SizedBox(height: 6),
                    const Text(
                      'PAUSED',
                      style: TextStyle(
                        fontSize: 22,
                        fontWeight: FontWeight.w900,
                        color: Color(0xFFFFB300),
                        letterSpacing: 1.5,
                      ),
                    ),
                    const Text(
                      'COVER CAMERA & FLASH',
                      style: TextStyle(
                        fontSize: 10,
                        fontWeight: FontWeight.w700,
                        color: Color(0xFFCBD5E1),
                        letterSpacing: 1.0,
                      ),
                    ),
                  ] else ...[
                    Text(
                      '$_secondsRemaining',
                      style: const TextStyle(
                        fontSize: 54,
                        fontWeight: FontWeight.w900,
                        color: Colors.white,
                        fontFeatures: [],
                      ),
                    ),
                    Text(
                      _state == ScannerState.measuring ? 'SECONDS LEFT' : '45 SEC TEST',
                      style: const TextStyle(
                        fontSize: 11,
                        fontWeight: FontWeight.w700,
                        letterSpacing: 1.5,
                        color: Color(0xFF94A3B8),
                      ),
                    ),
                  ],
                ],
              ),
            ),
          ],
        ),
        const SizedBox(height: 32),

        // Real-time Pulse Waveform Visualizer
        Container(
          height: 130,
          decoration: BoxDecoration(
            color: const Color(0xFF111827),
            borderRadius: BorderRadius.circular(16),
            border: Border.all(color: const Color(0xFF1F2937)),
          ),
          child: ClipRRect(
            borderRadius: BorderRadius.circular(16),
            child: AnimatedBuilder(
              animation: _animController,
              builder: (context, child) {
                return CustomPaint(
                  painter: PulseWaveformPainter(
                    values: _displayWaveformBuffer,
                    primaryColor: _isPausedDueToFingerLoss
                        ? const Color(0xFFFFB300)
                        : (_isFingerCoveringLens ? const Color(0xFF00E676) : const Color(0xFF00F0FF)),
                    isMeasuring: _state == ScannerState.measuring && !_isPausedDueToFingerLoss,
                    animationPhase: _animController.value,
                  ),
                  child: Container(),
                );
              },
            ),
          ),
        ),
        const SizedBox(height: 28),

        // Action Trigger Button
        if (_state == ScannerState.idle)
          ElevatedButton(
            onPressed: _startMeasurementFlow,
            style: ElevatedButton.styleFrom(
              backgroundColor: const Color(0xFF00E676),
              foregroundColor: Colors.black,
              minimumSize: const Size.fromHeight(56),
              shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(14)),
              elevation: 4,
            ),
            child: const Row(
              mainAxisAlignment: MainAxisAlignment.center,
              children: [
                Icon(Icons.play_arrow_rounded, size: 26),
                SizedBox(width: 8),
                Text(
                  'INITIATE 45s SCAN',
                  style: TextStyle(
                    fontSize: 15,
                    fontWeight: FontWeight.w800,
                    letterSpacing: 1.0,
                  ),
                ),
              ],
            ),
          )
        else if (_state == ScannerState.measuring || _state == ScannerState.fingerDetecting)
          OutlinedButton(
            onPressed: _resetToIdle,
            style: OutlinedButton.styleFrom(
              foregroundColor: const Color(0xFFFF5252),
              side: const BorderSide(color: Color(0xFFFF5252)),
              minimumSize: const Size.fromHeight(52),
              shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(14)),
            ),
            child: const Text(
              'CANCEL MEASUREMENT',
              style: TextStyle(fontWeight: FontWeight.w700, letterSpacing: 0.8),
            ),
          ),
      ],
    );
  }

  Widget _buildResultsView() {
    final result = _analysisResult!;
    final tierColor = result.tierColor;

    return Column(
      crossAxisAlignment: CrossAxisAlignment.stretch,
      children: [
        // Status Tier Badge Banner
        Container(
          padding: const EdgeInsets.symmetric(horizontal: 20, vertical: 16),
          decoration: BoxDecoration(
            color: tierColor.withValues(alpha: 0.12),
            borderRadius: BorderRadius.circular(18),
            border: Border.all(color: tierColor.withValues(alpha: 0.6), width: 1.5),
          ),
          child: Row(
            children: [
              Container(
                padding: const EdgeInsets.all(10),
                decoration: BoxDecoration(
                  color: tierColor,
                  shape: BoxShape.circle,
                ),
                child: Icon(result.tierIcon, color: Colors.black, size: 24),
              ),
              const SizedBox(width: 16),
              Expanded(
                child: Column(
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: [
                    Text(
                      'TIER: ${result.recoveryTier} READINESS',
                      style: TextStyle(
                        fontSize: 12,
                        fontWeight: FontWeight.w800,
                        color: tierColor,
                        letterSpacing: 1.3,
                      ),
                    ),
                    const SizedBox(height: 2),
                    Text(
                      result.stateLabel,
                      style: const TextStyle(
                        fontSize: 17,
                        fontWeight: FontWeight.w700,
                        color: Colors.white,
                      ),
                    ),
                  ],
                ),
              ),
            ],
          ),
        ),
        const SizedBox(height: 14),

        // Neuromuscular Readiness Score Gauge
        Container(
          padding: const EdgeInsets.symmetric(horizontal: 16, vertical: 12),
          decoration: BoxDecoration(
            color: const Color(0xFF131B2E),
            borderRadius: BorderRadius.circular(14),
            border: Border.all(color: const Color(0xFF1F2D4A)),
          ),
          child: Column(
            crossAxisAlignment: CrossAxisAlignment.start,
            children: [
              Row(
                mainAxisAlignment: MainAxisAlignment.spaceBetween,
                children: [
                  const Text(
                    'NEUROMUSCULAR READINESS',
                    style: TextStyle(
                      fontSize: 10,
                      fontWeight: FontWeight.w800,
                      color: Color(0xFF94A3B8),
                      letterSpacing: 1.1,
                    ),
                  ),
                  Text(
                    '${result.readinessScore.toStringAsFixed(1)}%',
                    style: TextStyle(
                      fontSize: 13,
                      fontWeight: FontWeight.w900,
                      color: tierColor,
                    ),
                  ),
                ],
              ),
              const SizedBox(height: 8),
              ClipRRect(
                borderRadius: BorderRadius.circular(6),
                child: LinearProgressIndicator(
                  value: (result.readinessScore / 100.0).clamp(0.0, 1.0),
                  minHeight: 6,
                  backgroundColor: const Color(0xFF1E293B),
                  valueColor: AlwaysStoppedAnimation<Color>(tierColor),
                ),
              ),
            ],
          ),
        ),
        const SizedBox(height: 14),

        // Primary Metrics: Heart Rate & rMSSD
        Row(
          children: [
            Expanded(
              child: _buildMetricTile(
                label: 'HEART RATE',
                value: result.heartRateBpm.toStringAsFixed(1),
                unit: 'BPM',
                icon: Icons.favorite_rounded,
                accentColor: const Color(0xFFFF5252),
              ),
            ),
            const SizedBox(width: 14),
            Expanded(
              child: _buildMetricTile(
                label: 'rMSSD (HRV)',
                value: result.rmssdMs.toStringAsFixed(1),
                unit: 'ms',
                icon: Icons.waves_rounded,
                accentColor: const Color(0xFF00F0FF),
              ),
            ),
          ],
        ),
        const SizedBox(height: 18),

        // Training Volume Prescription Card
        Container(
          padding: const EdgeInsets.all(18),
          decoration: BoxDecoration(
            color: const Color(0xFF131B2E),
            borderRadius: BorderRadius.circular(18),
            border: Border.all(color: const Color(0xFF1F2D4A)),
          ),
          child: Column(
            crossAxisAlignment: CrossAxisAlignment.start,
            children: [
              Row(
                mainAxisAlignment: MainAxisAlignment.spaceBetween,
                children: [
                  const Text(
                    'PRESCRIPTION',
                    style: TextStyle(
                      fontSize: 11,
                      fontWeight: FontWeight.w800,
                      color: Color(0xFF94A3B8),
                      letterSpacing: 1.3,
                    ),
                  ),
                  Container(
                    padding: const EdgeInsets.symmetric(horizontal: 10, vertical: 4),
                    decoration: BoxDecoration(
                      color: tierColor.withValues(alpha: 0.18),
                      borderRadius: BorderRadius.circular(20),
                    ),
                    child: Text(
                      result.volumeAdjustment,
                      style: TextStyle(
                        fontSize: 11,
                        fontWeight: FontWeight.w700,
                        color: tierColor,
                      ),
                    ),
                  ),
                ],
              ),
              const SizedBox(height: 12),
              Text(
                result.trainingPrescription,
                style: const TextStyle(
                  fontSize: 14,
                  height: 1.45,
                  color: Color(0xFFE2E8F0),
                  fontWeight: FontWeight.w500,
                ),
              ),
              const Divider(color: Color(0xFF1E293B), height: 28),
              const Text(
                'JOINT & CNS ADVISORY',
                style: TextStyle(
                  fontSize: 11,
                  fontWeight: FontWeight.w800,
                  color: Color(0xFF94A3B8),
                  letterSpacing: 1.2,
                ),
              ),
              const SizedBox(height: 6),
              Text(
                result.jointHealthAdvisory,
                style: const TextStyle(
                  fontSize: 13,
                  height: 1.4,
                  color: Color(0xFF94A3B8),
                ),
              ),
            ],
          ),
        ),
        const SizedBox(height: 18),

        // Recommended Activities Chips
        if (result.recommendedActivities.isNotEmpty) ...[
          const Text(
            'RECOMMENDED PROTOCOL',
            style: TextStyle(
              fontSize: 11,
              fontWeight: FontWeight.w800,
              color: Color(0xFF64748B),
              letterSpacing: 1.3,
            ),
          ),
          const SizedBox(height: 10),
          ...result.recommendedActivities.map(
            (act) => Padding(
              padding: const EdgeInsets.only(bottom: 8.0),
              child: Row(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  Icon(Icons.check_circle_rounded, size: 16, color: tierColor),
                  const SizedBox(width: 10),
                  Expanded(
                    child: Text(
                      act,
                      style: const TextStyle(
                        fontSize: 13,
                        color: Color(0xFFCBD5E1),
                        fontWeight: FontWeight.w500,
                      ),
                    ),
                  ),
                ],
              ),
            ),
          ),
          const SizedBox(height: 20),
        ],

        // Recalibrate / Start New Scan Button
        ElevatedButton(
          onPressed: _resetToIdle,
          style: ElevatedButton.styleFrom(
            backgroundColor: const Color(0xFF1E293B),
            foregroundColor: Colors.white,
            minimumSize: const Size.fromHeight(54),
            shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(14)),
          ),
          child: const Row(
            mainAxisAlignment: MainAxisAlignment.center,
            children: [
              Icon(Icons.refresh_rounded, size: 20),
              SizedBox(width: 8),
              Text(
                'RECORD NEW SESSION',
                style: TextStyle(fontWeight: FontWeight.w700, letterSpacing: 0.8),
              ),
            ],
          ),
        ),
      ],
    );
  }

  Widget _buildMetricTile({
    required String label,
    required String value,
    required String unit,
    required IconData icon,
    required Color accentColor,
  }) {
    return Container(
      padding: const EdgeInsets.all(16),
      decoration: BoxDecoration(
        color: const Color(0xFF131B2E),
        borderRadius: BorderRadius.circular(16),
        border: Border.all(color: const Color(0xFF1F2D4A)),
      ),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Row(
            mainAxisAlignment: MainAxisAlignment.spaceBetween,
            children: [
              Text(
                label,
                style: const TextStyle(
                  fontSize: 10,
                  fontWeight: FontWeight.w800,
                  color: Color(0xFF94A3B8),
                  letterSpacing: 1.1,
                ),
              ),
              Icon(icon, size: 16, color: accentColor),
            ],
          ),
          const SizedBox(height: 10),
          Row(
            crossAxisAlignment: CrossAxisAlignment.baseline,
            textBaseline: TextBaseline.alphabetic,
            children: [
              Text(
                value,
                style: const TextStyle(
                  fontSize: 28,
                  fontWeight: FontWeight.w900,
                  color: Colors.white,
                ),
              ),
              const SizedBox(width: 4),
              Text(
                unit,
                style: TextStyle(
                  fontSize: 12,
                  fontWeight: FontWeight.w700,
                  color: accentColor,
                ),
              ),
            ],
          ),
        ],
      ),
    );
  }

  Widget _buildErrorBanner() {
    return Container(
      margin: const EdgeInsets.only(bottom: 16),
      padding: const EdgeInsets.all(14),
      decoration: BoxDecoration(
        color: const Color(0xFFFF1744).withValues(alpha: 0.12),
        borderRadius: BorderRadius.circular(12),
        border: Border.all(color: const Color(0xFFFF1744).withValues(alpha: 0.5)),
      ),
      child: Row(
        children: [
          const Icon(Icons.error_outline_rounded, color: Color(0xFFFF1744), size: 22),
          const SizedBox(width: 12),
          Expanded(
            child: Text(
              _errorMessage!,
              style: const TextStyle(fontSize: 13, color: Color(0xFFFF8A80), height: 1.3),
            ),
          ),
          IconButton(
            icon: const Icon(Icons.close_rounded, size: 18, color: Colors.white70),
            onPressed: () => setState(() => _errorMessage = null),
          ),
        ],
      ),
    );
  }

  void _showSettingsDialog() {
    final controller = TextEditingController(text: _apiService.baseUrl);

    showDialog(
      context: context,
      builder: (ctx) => AlertDialog(
        backgroundColor: const Color(0xFF131B2E),
        title: const Text('Backend API Host', style: TextStyle(color: Colors.white, fontSize: 16)),
        content: Column(
          mainAxisSize: MainAxisSize.min,
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            const Text(
              'Set FastAPI endpoint URL:',
              style: TextStyle(color: Color(0xFF94A3B8), fontSize: 13),
            ),
            const SizedBox(height: 10),
            TextField(
              controller: controller,
              style: const TextStyle(color: Colors.white, fontSize: 14),
              decoration: InputDecoration(
                filled: true,
                fillColor: const Color(0xFF0F172A),
                hintText: 'http://10.0.2.2:8000',
                hintStyle: const TextStyle(color: Color(0xFF475569)),
                border: OutlineInputBorder(borderRadius: BorderRadius.circular(8)),
              ),
            ),
            const SizedBox(height: 12),
            const Text(
              '• Android Emulator: http://10.0.2.2:8000\n'
              '• Physical Device: http://<YOUR_PC_LAN_IP>:8000\n'
              '• Web / iOS: http://127.0.0.1:8000',
              style: TextStyle(color: Color(0xFF64748B), fontSize: 11, height: 1.4),
            ),
          ],
        ),
        actions: [
          TextButton(
            onPressed: () => Navigator.pop(ctx),
            child: const Text('Cancel', style: TextStyle(color: Color(0xFF94A3B8))),
          ),
          ElevatedButton(
            onPressed: () {
              setState(() {
                _apiService.baseUrl = controller.text.trim();
              });
              Navigator.pop(ctx);
            },
            style: ElevatedButton.styleFrom(backgroundColor: const Color(0xFF00E676)),
            child: const Text('Save', style: TextStyle(color: Colors.black)),
          ),
        ],
      ),
    );
  }
}

class _OpticalMetrics {
  final double luminance;
  final bool isFingerCovering;
  final double redChroma;
  final double blueChroma;
  final double chromaDiff;

  _OpticalMetrics({
    required this.luminance,
    required this.isFingerCovering,
    this.redChroma = 0.0,
    this.blueChroma = 0.0,
    this.chromaDiff = 0.0,
  });
}
