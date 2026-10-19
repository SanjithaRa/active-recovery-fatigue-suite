import 'package:flutter_test/flutter_test.dart';
import 'package:recovery_mobile/main.dart';

void main() {
  testWidgets('ActiveRecoveryApp smoke test', (WidgetTester tester) async {
    await tester.pumpWidget(const ActiveRecoveryApp());
    expect(find.byType(ActiveRecoveryApp), findsOneWidget);
  });
}
