import 'package:flutter/material.dart';

/// Models the autonomic fatigue recovery analysis returned from the FastAPI backend.
class RecoveryResult {
  final double heartRateBpm;
  final double rmssdMs;
  final String recoveryTier; // "GREEN" | "YELLOW" | "RED"
  final String tierColorHex;
  final String stateLabel;
  final double readinessScore; // 0 - 100%
  final String volumeAdjustment;
  final String cnsReadiness;
  final String trainingPrescription;
  final List<String> recommendedActivities;
  final String jointHealthAdvisory;
  final String autonomicSummary;
  final double greenProb;
  final double yellowProb;
  final double redProb;
  final String inferenceEngine;
  final double signalQualityScore;
  final double samplingRateHz;
  final double durationSeconds;
  final int peakCount;
  final List<double> rrIntervalsMs;
  final String timestamp;

  const RecoveryResult({
    required this.heartRateBpm,
    required this.rmssdMs,
    required this.recoveryTier,
    required this.tierColorHex,
    required this.stateLabel,
    required this.readinessScore,
    required this.volumeAdjustment,
    required this.cnsReadiness,
    required this.trainingPrescription,
    required this.recommendedActivities,
    required this.jointHealthAdvisory,
    required this.autonomicSummary,
    required this.greenProb,
    required this.yellowProb,
    required this.redProb,
    required this.inferenceEngine,
    required this.signalQualityScore,
    required this.samplingRateHz,
    required this.durationSeconds,
    required this.peakCount,
    required this.rrIntervalsMs,
    required this.timestamp,
  });

  /// Factory deserializer from backend JSON
  factory RecoveryResult.fromJson(Map<String, dynamic> json) {
    final probs = json['model_probabilities'] as Map<String, dynamic>?;

    return RecoveryResult(
      heartRateBpm: (json['heart_rate_bpm'] as num?)?.toDouble() ?? 0.0,
      rmssdMs: (json['rmssd_ms'] as num?)?.toDouble() ?? 0.0,
      recoveryTier: json['recovery_tier'] as String? ?? 'YELLOW',
      tierColorHex: json['tier_color_hex'] as String? ?? '#FFD600',
      stateLabel: json['state_label'] as String? ?? 'Moderate Autonomic State',
      readinessScore: (json['readiness_score'] as num?)?.toDouble() ??
          (json['recovery_tier'] == 'GREEN'
              ? 92.0
              : (json['recovery_tier'] == 'RED' ? 32.0 : 68.0)),
      volumeAdjustment: json['volume_adjustment'] as String? ?? 'Standard Load',
      cnsReadiness: json['cns_readiness'] as String? ?? 'Moderate CNS Capacity',
      trainingPrescription: json['training_prescription'] as String? ?? 'Proceed with caution.',
      recommendedActivities: (json['recommended_activities'] as List<dynamic>?)
              ?.map((e) => e.toString())
              .toList() ??
          [],
      jointHealthAdvisory: json['joint_health_advisory'] as String? ??
          'Maintain joint mobility and warm-up.',
      autonomicSummary: json['autonomic_summary'] as String? ?? '',
      greenProb: (probs?['green_prob'] as num?)?.toDouble() ?? 0.0,
      yellowProb: (probs?['yellow_prob'] as num?)?.toDouble() ?? 0.0,
      redProb: (probs?['red_prob'] as num?)?.toDouble() ?? 0.0,
      inferenceEngine: json['inference_engine'] as String? ?? 'Production-ML',
      signalQualityScore: (json['signal_quality_score'] as num?)?.toDouble() ?? 80.0,
      samplingRateHz: (json['sampling_rate_hz'] as num?)?.toDouble() ?? 30.0,
      durationSeconds: (json['duration_seconds'] as num?)?.toDouble() ?? 45.0,
      peakCount: json['peak_count'] as int? ?? 0,
      rrIntervalsMs: (json['rr_intervals_ms'] as List<dynamic>?)
              ?.map((e) => (e as num).toDouble())
              .toList() ??
          [],
      timestamp: json['timestamp'] as String? ?? DateTime.now().toIso8601String(),
    );
  }

  /// Convert hex string (#00E676) to Flutter Color
  Color get tierColor {
    try {
      final hexClean = tierColorHex.replaceFirst('#', '');
      return Color(int.parse('FF$hexClean', radix: 16));
    } catch (_) {
      switch (recoveryTier.toUpperCase()) {
        case 'GREEN':
          return const Color(0xFF00E676);
        case 'RED':
          return const Color(0xFFFF1744);
        default:
          return const Color(0xFFFFD600);
      }
    }
  }

  /// Icon corresponding to recovery status
  IconData get tierIcon {
    switch (recoveryTier.toUpperCase()) {
      case 'GREEN':
        return Icons.bolt_rounded;
      case 'RED':
        return Icons.healing_rounded;
      default:
        return Icons.speed_rounded;
    }
  }

  bool get isOptimal => recoveryTier.toUpperCase() == 'GREEN';
  bool get isFatigued => recoveryTier.toUpperCase() == 'RED';
}
