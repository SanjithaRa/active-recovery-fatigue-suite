import 'dart:async';
import 'dart:convert';
import 'dart:io';
import 'package:http/http.dart' as http;
import '../models/recovery_result.dart';

class ApiException implements Exception {
  final String message;
  final String? errorCode;
  final String? recommendation;

  ApiException(this.message, {this.errorCode, this.recommendation});

  @override
  String toString() => message;
}

class FatigueApiService {
  /// Default base URL configuration:
  /// - Physical device over USB (adb reverse): http://127.0.0.1:8000
  /// - Wi-Fi LAN: http://10.64.198.90:8000
  /// - Android Emulator: http://10.0.2.2:8000
  static String defaultBaseUrl() {
    return 'http://127.0.0.1:8000';
  }

  String baseUrl;

  FatigueApiService({String? initialBaseUrl})
      : baseUrl = initialBaseUrl ?? defaultBaseUrl();

  /// Post optical time-series stream to backend DSP & rules pipeline
  /// Automatically tries candidate URLs if primary host is unreachable
  Future<RecoveryResult> analyzeFatigue({
    required List<double> timestampsMs,
    required List<double> rawSignals,
    String? subjectId,
  }) async {
    final payload = {
      'timestamps_ms': timestampsMs,
      'raw_signals': rawSignals,
      'device_info': 'Flutter Mobile Camera PPG (Android/iOS)',
      'subject_id': subjectId ?? 'athlete_primary',
    };

    // Candidates in priority order
    final candidateHosts = <String>{
      baseUrl,
      'http://127.0.0.1:8000',
      'http://10.64.198.90:8000',
      'http://10.0.2.2:8000',
    }.toList();

    String? lastErrorMsg;

    for (final host in candidateHosts) {
      try {
        final uri = Uri.parse('$host/api/analyze-fatigue');
        final response = await http
            .post(
              uri,
              headers: {
                'Content-Type': 'application/json',
                'Accept': 'application/json',
              },
              body: jsonEncode(payload),
            )
            .timeout(const Duration(seconds: 10));

        final Map<String, dynamic> data = jsonDecode(response.body);

        if (response.statusCode >= 200 && response.statusCode < 300) {
          // Cache successful host for future calls
          baseUrl = host;
          return RecoveryResult.fromJson(data);
        } else {
          final message = data['message'] ?? data['detail'] ?? 'Analysis failed';
          final errorCode = data['error_code'];
          final recommendation = data['recommendation'];
          throw ApiException(
            message.toString(),
            errorCode: errorCode?.toString(),
            recommendation: recommendation?.toString(),
          );
        }
      } on SocketException catch (e) {
        lastErrorMsg = 'Cannot reach $host (${e.message})';
        continue;
      } on TimeoutException {
        lastErrorMsg = 'Connection to $host timed out';
        continue;
      } on FormatException catch (e) {
        throw ApiException(
          'Failed to parse backend response: ${e.message}',
          errorCode: 'INVALID_RESPONSE_FORMAT',
        );
      } catch (e) {
        if (e is ApiException) rethrow;
        lastErrorMsg = e.toString();
        continue;
      }
    }

    throw ApiException(
      'Could not connect to FastAPI backend at any known address.\n'
      'Last error: $lastErrorMsg\n\n'
      'Ensure the server is running on your PC and tap the top-right settings icon to check the host URL.',
      errorCode: 'ALL_HOSTS_UNREACHABLE',
      recommendation: 'Check that USB cable is plugged in or Wi-Fi IP is correct.',
    );
  }

  Future<bool> checkHealth() async {
    try {
      final response = await http
          .get(Uri.parse('$baseUrl/health'))
          .timeout(const Duration(seconds: 4));
      return response.statusCode == 200;
    } catch (_) {
      return false;
    }
  }
}
