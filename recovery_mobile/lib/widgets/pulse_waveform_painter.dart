import 'dart:math' as math;
import 'package:flutter/material.dart';

/// CustomPainter that renders a smooth, real-time glowing photoplethysmogram (PPG)
/// pulse waveform with athletic dark-mode gridlines and gradient area fill.
class PulseWaveformPainter extends CustomPainter {
  final List<double> values;
  final Color primaryColor;
  final bool isMeasuring;
  final double animationPhase;

  PulseWaveformPainter({
    required this.values,
    required this.primaryColor,
    this.isMeasuring = false,
    this.animationPhase = 0.0,
  });

  @override
  void paint(Canvas canvas, Size size) {
    _drawAthleticGrid(canvas, size);

    if (values.length < 2) {
      _drawIdleGuideline(canvas, size);
      return;
    }

    // Dynamic min-max scaling with baseline drift isolation
    double minVal = values.first;
    double maxVal = values.first;
    for (final v in values) {
      if (v < minVal) minVal = v;
      if (v > maxVal) maxVal = v;
    }

    double range = maxVal - minVal;
    if (range < 0.001) {
      range = 1.0; // avoid divide-by-zero on flatline
    }

    // Padding bounds for aesthetic waveform clearance
    const double verticalPadding = 20.0;
    final double usableHeight = size.height - (verticalPadding * 2);

    // Compute coordinate points
    final List<Offset> points = [];
    final double stepX = size.width / (values.length - 1);

    for (int i = 0; i < values.length; i++) {
      final double normalized = (values[i] - minVal) / range;
      // Invert Y so high optical reflection is at the top
      final double y = size.height - verticalPadding - (normalized * usableHeight);
      final double x = i * stepX;
      points.add(Offset(x, y));
    }

    // Build smooth cubic bezier spline through points
    final Path wavePath = Path();
    wavePath.moveTo(points.first.dx, points.first.dy);

    for (int i = 0; i < points.length - 1; i++) {
      final p0 = points[i];
      final p1 = points[i + 1];
      final controlPointX1 = p0.dx + (p1.dx - p0.dx) / 2.0;
      final controlPointX2 = controlPointX1;
      wavePath.cubicTo(
        controlPointX1,
        p0.dy,
        controlPointX2,
        p1.dy,
        p1.dx,
        p1.dy,
      );
    }

    // 1. Draw gradient fill under the pulse wave
    final Path fillPath = Path.from(wavePath)
      ..lineTo(points.last.dx, size.height)
      ..lineTo(points.first.dx, size.height)
      ..close();

    final fillGradient = LinearGradient(
      begin: Alignment.topCenter,
      end: Alignment.bottomCenter,
      colors: [
        primaryColor.withValues(alpha: 0.35),
        primaryColor.withValues(alpha: 0.08),
        Colors.transparent,
      ],
      stops: const [0.0, 0.6, 1.0],
    );

    final fillPaint = Paint()
      ..shader = fillGradient.createShader(Rect.fromLTWH(0, 0, size.width, size.height))
      ..style = PaintingStyle.fill;

    canvas.drawPath(fillPath, fillPaint);

    // 2. Draw outer neon glow pass
    final glowPaint = Paint()
      ..color = primaryColor.withValues(alpha: 0.35)
      ..strokeWidth = 6.0
      ..strokeCap = StrokeCap.round
      ..style = PaintingStyle.stroke
      ..maskFilter = const MaskFilter.blur(BlurStyle.normal, 4.0);

    canvas.drawPath(wavePath, glowPaint);

    // 3. Draw sharp primary trace
    final strokePaint = Paint()
      ..color = primaryColor
      ..strokeWidth = 2.4
      ..strokeCap = StrokeCap.round
      ..strokeJoin = StrokeJoin.round
      ..style = PaintingStyle.stroke;

    canvas.drawPath(wavePath, strokePaint);

    // 4. Draw active leading scanhead pulse dot at the latest sample
    if (isMeasuring && points.isNotEmpty) {
      final leadingPoint = points.last;
      final pulseRadius = 5.0 + 2.0 * math.sin(animationPhase * 2 * math.pi);

      final haloPaint = Paint()
        ..color = primaryColor.withValues(alpha: 0.4)
        ..style = PaintingStyle.fill;
      canvas.drawCircle(leadingPoint, pulseRadius + 4, haloPaint);

      final dotPaint = Paint()
        ..color = Colors.white
        ..style = PaintingStyle.fill;
      canvas.drawCircle(leadingPoint, pulseRadius, dotPaint);
    }
  }

  void _drawAthleticGrid(Canvas canvas, Size size) {
    final gridPaint = Paint()
      ..color = const Color(0xFF1E293B).withValues(alpha: 0.6)
      ..strokeWidth = 1.0;

    // Horizontal grid guidelines
    const int horizontalDivisions = 4;
    for (int i = 1; i < horizontalDivisions; i++) {
      final y = size.height * (i / horizontalDivisions);
      canvas.drawLine(Offset(0, y), Offset(size.width, y), gridPaint);
    }

    // Vertical time guidelines
    const int verticalDivisions = 6;
    for (int i = 1; i < verticalDivisions; i++) {
      final x = size.width * (i / verticalDivisions);
      canvas.drawLine(Offset(x, 0), Offset(x, size.height), gridPaint);
    }
  }

  void _drawIdleGuideline(Canvas canvas, Size size) {
    final midY = size.height / 2;
    final idlePaint = Paint()
      ..color = const Color(0xFF334155)
      ..strokeWidth = 1.5
      ..style = PaintingStyle.stroke;

    // Subtle dashed center idle line
    const double dashWidth = 6.0;
    const double dashSpace = 4.0;
    double startX = 0.0;
    while (startX < size.width) {
      canvas.drawLine(
        Offset(startX, midY),
        Offset(math.min(startX + dashWidth, size.width), midY),
        idlePaint,
      );
      startX += dashWidth + dashSpace;
    }
  }

  @override
  bool shouldRepaint(covariant PulseWaveformPainter oldDelegate) {
    return oldDelegate.values != values ||
        oldDelegate.primaryColor != primaryColor ||
        oldDelegate.isMeasuring != isMeasuring ||
        oldDelegate.animationPhase != animationPhase;
  }
}
