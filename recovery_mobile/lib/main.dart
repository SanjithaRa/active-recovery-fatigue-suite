import 'package:camera/camera.dart';
import 'package:flutter/material.dart';
import 'package:flutter/services.dart';
import 'screens/pulse_scanner_screen.dart';

List<CameraDescription> _availableCameras = [];

Future<void> main() async {
  WidgetsFlutterBinding.ensureInitialized();

  // Set immersive dark status bar and navigation bar styling
  SystemChrome.setSystemUIOverlayStyle(
    const SystemUiOverlayStyle(
      statusBarColor: Colors.transparent,
      statusBarIconBrightness: Brightness.light,
      systemNavigationBarColor: Color(0xFF0A0E17),
      systemNavigationBarIconBrightness: Brightness.light,
    ),
  );

  // Discover connected camera hardware
  try {
    _availableCameras = await availableCameras();
  } catch (e) {
    debugPrint('Unable to query device cameras at startup: $e');
    _availableCameras = [];
  }

  runApp(const ActiveRecoveryApp());
}

class ActiveRecoveryApp extends StatelessWidget {
  const ActiveRecoveryApp({super.key});

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'Active Recovery Suite',
      debugShowCheckedModeBanner: false,
      themeMode: ThemeMode.dark,
      darkTheme: ThemeData(
        useMaterial3: true,
        brightness: Brightness.dark,
        scaffoldBackgroundColor: const Color(0xFF0A0E17),
        colorScheme: const ColorScheme.dark(
          primary: Color(0xFF00E676),
          secondary: Color(0xFF00F0FF),
          tertiary: Color(0xFFFFD600),
          error: Color(0xFFFF1744),
          surface: Color(0xFF101726),
          onSurface: Colors.white,
        ),
        fontFamily: 'Roboto',
        cardTheme: const CardThemeData(
          color: Color(0xFF131B2E),
          elevation: 0,
        ),
        appBarTheme: const AppBarTheme(
          backgroundColor: Color(0xFF101726),
          elevation: 0,
          scrolledUnderElevation: 0,
          centerTitle: false,
        ),
      ),
      home: PulseScannerScreen(cameras: _availableCameras),
    );
  }
}
