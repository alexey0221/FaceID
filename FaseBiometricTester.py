# import cv2
# import numpy as np
# import json
# import os
# import sys
# from pathlib import Path
# from datetime import datetime
# import matplotlib.pyplot as plt
# from collections import defaultdict
#
# # Добавляем путь к родительской директории для импорта основного класса
# sys.path.append(os.path.dirname(os.path.abspath(__file__)))
#
#
# # Импортируем ваш класс (предполагается, что он в отдельном файле)
# # Если класс в этом же файле, то импорт не нужен
# # from FaceBiometricSystem import FaceBiometricSystem
# from FaseBiometricSystem import FaceBiometricSystem
#
# class FaceBiometricTester:
#     """Класс для тестирования биометрической системы"""
#
#     def __init__(self, system, people_dir="Люди"):
#         """
#         Инициализация тестера
#
#         Args:
#             system: экземпляр FaceBiometricSystem
#             people_dir: директория с папками 1 и 2
#         """
#         self.system = system
#         self.people_dir = Path(people_dir)
#         self.train_dir = self.people_dir / "1"
#         self.test_dir = self.people_dir / "2"
#
#         # Результаты тестирования
#         self.results = {
#             'authentications': [],
#             'statistics': {},
#             'errors': [],
#             'confusion_matrix': defaultdict(lambda: defaultdict(int))
#         }
#
#     def get_people_list(self):
#         """Получает список людей из папки 1"""
#         people = []
#         if self.train_dir.exists():
#             for file in self.train_dir.iterdir():
#                 if file.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp']:
#                     name = file.stem  # Имя без расширения
#                     people.append(name)
#         return people
#
#     def register_all_users(self):
#         """Регистрирует всех пользователей из папки 1"""
#         print("\n" + "=" * 70)
#         print("РЕГИСТРАЦИЯ ПОЛЬЗОВАТЕЛЕЙ")
#         print("=" * 70)
#
#         people = self.get_people_list()
#
#         if not people:
#             print("❌ Ошибка: Не найдены изображения в папке 'Люди/1'")
#             return False
#
#         print(f"Найдено {len(people)} пользователей для регистрации:")
#         for i, person in enumerate(people, 1):
#             print(f"  {i}. {person}")
#
#         successful = []
#         failed = []
#
#         for person in people:
#             image_path = self.train_dir / f"{person}.jpg"
#             # Проверяем другие расширения
#             if not os.path.exists(image_path):#.exists():
#                 for ext in ['.jpeg', '.png', '.bmp']:
#                     alt_path = self.train_dir / f"{person}{ext}"
#                     if alt_path.exists():
#                         image_path = alt_path
#                         break
#
#             print(f"\n📝 Регистрация: {person}")
#             if self.system.register_user(person, "file", str(image_path)):
#                 successful.append(person)
#             else:
#                 failed.append(person)
#
#         print("\n" + "=" * 70)
#         print("РЕЗУЛЬТАТЫ РЕГИСТРАЦИИ")
#         print("=" * 70)
#         print(f"✅ Успешно: {len(successful)}/{len(people)}")
#         if failed:
#             print(f"❌ Неудачно: {failed}")
#
#         return len(successful) > 0
#
#     def test_all_users(self, tolerance_values=None):
#         """
#         Тестирует всех пользователей с разными порогами ошибки
#
#         Args:
#             tolerance_values: список значений порога для тестирования
#         """
#         if tolerance_values is None:
#             tolerance_values = [0.03, 0.05, 0.07, 0.10, 0.12, 0.15]
#
#         print("\n" + "=" * 70)
#         print("ТЕСТИРОВАНИЕ СИСТЕМЫ")
#         print("=" * 70)
#
#         people = self.get_people_list()
#
#         if not people:
#             print("❌ Ошибка: Нет зарегистрированных пользователей")
#             return
#
#         # Результаты для разных порогов
#         all_results = {}
#
#         for tolerance in tolerance_values:
#             print(f"\n📊 Тестирование с порогом ошибки: {tolerance * 100:.1f}%")
#             print("-" * 50)
#
#             self.system.tolerance = tolerance
#             results = self.run_tests(people)
#             all_results[tolerance] = results
#
#             self.print_summary(results)
#
#         # Анализ и оптимизация
#         self.analyze_and_optimize(all_results, people)
#
#         return all_results
#
#     def run_tests(self, people):
#         """Запускает тесты для всех пользователей"""
#         results = {
#             'total': 0,
#             'correct': 0,
#             'false_positive': 0,
#             'false_negative': 0,
#             'details': [],
#             'scores': {}
#         }
#
#         for person in people:
#             # Находим тестовое изображение
#             test_image = self.test_dir / f"{person}.jpg"
#             if not test_image.exists():
#                 for ext in ['.jpeg', '.png', '.bmp']:
#                     alt_path = self.test_dir / f"{person}{ext}"
#                     if alt_path.exists():
#                         test_image = alt_path
#                         break
#
#             if not test_image.exists():
#                 print(f"⚠️ Предупреждение: Нет тестового изображения для {person}")
#                 continue
#
#             print(f"\n🔐 Тестирование: {person}")
#
#             # Пытаемся аутентифицироваться как этот пользователь
#             auth_result = self.system.authenticate_user(person, "file", str(test_image))
#
#             is_correct = auth_result.get('authenticated', False)
#             similarity = auth_result.get('similarity', 0)
#             avg_error = auth_result.get('avg_error', 1)
#
#             result_detail = {
#                 'person': person,
#                 'authenticated': is_correct,
#                 'similarity': similarity,
#                 'avg_error': avg_error,
#                 'is_correct': is_correct  # Для правильного пользователя
#             }
#
#             results['details'].append(result_detail)
#             results['total'] += 1
#
#             if is_correct:
#                 results['correct'] += 1
#                 results['scores'][person] = similarity
#                 print(f"  ✅ Верно (схожесть: {similarity * 100:.1f}%)")
#             else:
#                 results['false_negative'] += 1
#                 print(f"  ❌ Ошибка FN (схожесть: {similarity * 100:.1f}%)")
#
#             # Проверяем cross-аутентификацию (false positives)
#             for other_person in people:
#                 if other_person != person:
#                     print(f"  🔄 Cross-test: {person} как {other_person}")
#                     cross_result = self.system.authenticate_user(other_person, "file", str(test_image))
#
#                     if cross_result.get('authenticated', False):
#                         results['false_positive'] += 1
#                         print(f"    ⚠️ Ложное срабатывание: {person} распознан как {other_person}")
#                         self.results['confusion_matrix'][person][other_person] += 1
#
#         return results
#
#     def print_summary(self, results):
#         """Выводит сводку результатов"""
#         print("\n" + "=" * 50)
#         print("СВОДКА ТЕСТИРОВАНИЯ")
#         print("=" * 50)
#
#         accuracy = results['correct'] / results['total'] if results['total'] > 0 else 0
#         far = results['false_positive'] / (results['total'] * (results['total'] - 1)) if results['total'] > 1 else 0
#         frr = results['false_negative'] / results['total'] if results['total'] > 0 else 0
#
#         print(f"📊 Общая статистика:")
#         print(f"  • Всего тестов: {results['total']}")
#         print(f"  • Верных: {results['correct']}")
#         print(f"  • Ложных отказов (FN): {results['false_negative']}")
#         print(f"  • Ложных срабатываний (FP): {results['false_positive']}")
#         print(f"\n📈 Метрики:")
#         print(f"  • Точность (Accuracy): {accuracy * 100:.2f}%")
#         print(f"  • FAR (False Acceptance Rate): {far * 100:.4f}%")
#         print(f"  • FRR (False Rejection Rate): {frr * 100:.2f}%")
#
#         if results['scores']:
#             avg_similarity = np.mean(list(results['scores'].values()))
#             print(f"  • Средняя схожесть: {avg_similarity * 100:.2f}%")
#
#     def analyze_and_optimize(self, all_results, people):
#         """
#         Анализирует результаты и предлагает оптимальные параметры
#
#         Args:
#             all_results: результаты для разных порогов
#             people: список пользователей
#         """
#         print("\n" + "=" * 70)
#         print("АНАЛИЗ И ОПТИМИЗАЦИЯ")
#         print("=" * 70)
#
#         # Находим оптимальный порог
#         best_tolerance = None
#         best_accuracy = 0
#         best_metrics = None
#
#         for tolerance, results in all_results.items():
#             accuracy = results['correct'] / results['total'] if results['total'] > 0 else 0
#
#             if accuracy > best_accuracy:
#                 best_accuracy = accuracy
#                 best_tolerance = tolerance
#                 best_metrics = results
#         if best_tolerance:
#             print(f"\n🎯 Оптимальный порог ошибки: {best_tolerance * 100:.1f}%")
#         if best_accuracy:
#             print(f"   Достигнутая точность: {best_accuracy * 100:.2f}%")
#
#         # Анализ ошибок
#         print(f"\n📋 Детальный анализ ошибок:")
#
#         if best_metrics:
#             # Анализ false negatives
#             fn_cases = [d for d in best_metrics['details'] if not d['is_correct']]
#             if fn_cases:
#                 print(f"\n  ❌ Ложные отказы (FN):")
#                 for case in fn_cases:
#                     print(f"    - {case['person']}: схожесть {case['similarity'] * 100:.1f}%, "
#                           f"ошибка {case['avg_error'] * 100:.1f}%")
#
#             # Анализ false positives из confusion matrix
#             if self.results['confusion_matrix']:
#                 print(f"\n  ⚠️ Ложные срабатывания (FP):")
#                 for real, fakes in self.results['confusion_matrix'].items():
#                     for fake, count in fakes.items():
#                         if count > 0:
#                             print(f"    - {real} распознан как {fake} ({count} раз)")
#
#         # Рекомендации по улучшению
#         print(f"\n💡 РЕКОМЕНДАЦИИ ПО УЛУЧШЕНИЮ:")
#
#         if best_accuracy < 0.8:
#             print(f"  1. 🔧 Текущая точность низкая ({best_accuracy * 100:.1f}%)")
#             print(f"     • Увеличьте количество точек лица (улучшите детекцию)")
#             print(f"     • Используйте более качественные изображения для регистрации")
#             print(f"     • Добавьте предобработку изображений (выравнивание, нормализацию)")
#
#         # Анализ вариативности
#         if best_metrics and best_metrics['scores']:
#             similarities = list(best_metrics['scores'].values())
#             variance = np.var(similarities)
#             if variance > 0.05:
#                 print(f"  2. 📊 Высокая вариативность схожести ({variance:.4f})")
#                 print(f"     • Некоторые пользователи распознаются хуже других")
#                 print(f"     • Рассмотрите индивидуальные пороги для сложных случаев")
#
#         # Рекомендации по порогу
#         print(f"  3. 🎛️ Настройка порога:")
#         if best_tolerance:
#             print(f"     • Текущий оптимальный: {best_tolerance * 100:.1f}%")
#
#         # Проверяем, нужно ли уменьшить FAR
#         if best_metrics:
#             far = best_metrics['false_positive'] / (best_metrics['total'] * (best_metrics['total'] - 1)) if \
#             best_metrics['total'] > 1 else 0
#             if far > 0.01:  # Больше 1%
#                 print(f"     • Высокий FAR ({far * 100:.2f}%), уменьшите порог до {best_tolerance * 100 - 1:.1f}%")
#             else:
#                 print(f"     • Хороший FAR ({far * 100:.2f}%)")
#
#     def plot_results(self, all_results):
#         """Визуализирует результаты тестирования"""
#         fig, axes = plt.subplots(2, 2, figsize=(12, 10))
#         fig.suptitle('Анализ биометрической системы', fontsize=16)
#
#         # 1. График точности от порога
#         ax1 = axes[0, 0]
#         tolerances = list(all_results.keys())
#         accuracies = [r['correct'] / r['total'] if r['total'] > 0 else 0 for r in all_results.values()]
#
#         ax1.plot(tolerances, accuracies, 'bo-', linewidth=2, markersize=8)
#         ax1.set_xlabel('Порог ошибки', fontsize=12)
#         ax1.set_ylabel('Точность', fontsize=12)
#         ax1.set_title('Зависимость точности от порога', fontsize=12)
#         ax1.grid(True, alpha=0.3)
#         ax1.set_xlim(min(tolerances) - 0.01, max(tolerances) + 0.01)
#         ax1.set_ylim(0, 1.05)
#
#         # 2. FAR и FRR
#         ax2 = axes[0, 1]
#         far_values = []
#         frr_values = []
#
#         for tolerance, results in all_results.items():
#             total = results['total']
#             far = results['false_positive'] / (total * (total - 1)) if total > 1 else 0
#             frr = results['false_negative'] / total if total > 0 else 0
#             far_values.append(far)
#             frr_values.append(frr)
#
#         ax2.plot(tolerances, far_values, 'r^-', linewidth=2, markersize=8, label='FAR')
#         ax2.plot(tolerances, frr_values, 'gs-', linewidth=2, markersize=8, label='FRR')
#         ax2.set_xlabel('Порог ошибки', fontsize=12)
#         ax2.set_ylabel('Ошибки', fontsize=12)
#         ax2.set_title('FAR и FRR', fontsize=12)
#         ax2.legend()
#         ax2.grid(True, alpha=0.3)
#
#         # 3. Распределение схожести
#         ax3 = axes[1, 0]
#         if all_results:
#             best_tolerance = max(all_results.keys(), key=lambda t: all_results[t]['correct'] / all_results[t]['total'])
#             similarities = [d['similarity'] for d in all_results[best_tolerance]['details']]
#
#             ax3.hist(similarities, bins=10, alpha=0.7, color='blue', edgecolor='black')
#             ax3.axvline(x=best_tolerance, color='red', linestyle='--',
#                         label=f'Порог: {best_tolerance * 100:.1f}%')
#             ax3.set_xlabel('Схожесть', fontsize=12)
#             ax3.set_ylabel('Частота', fontsize=12)
#             ax3.set_title('Распределение схожести', fontsize=12)
#             ax3.legend()
#             ax3.grid(True, alpha=0.3)
#
#         # 4. Confusion matrix heatmap (упрощенная)
#         ax4 = axes[1, 1]
#         if self.results['confusion_matrix']:
#             people = list(self.results['confusion_matrix'].keys())
#             matrix = np.zeros((len(people), len(people)))
#
#             for i, real in enumerate(people):
#                 for j, fake in enumerate(people):
#                     if i != j:
#                         matrix[i, j] = self.results['confusion_matrix'][real].get(fake, 0)
#
#             im = ax4.imshow(matrix, cmap='YlOrRd', interpolation='nearest')
#             ax4.set_xticks(range(len(people)))
#             ax4.set_yticks(range(len(people)))
#             ax4.set_xticklabels(people, rotation=45, ha='right', fontsize=8)
#             ax4.set_yticklabels(people, fontsize=8)
#             ax4.set_xlabel('Распознан как', fontsize=12)
#             ax4.set_ylabel('Реальный пользователь', fontsize=12)
#             ax4.set_title('Матрица ошибок', fontsize=12)
#             plt.colorbar(im, ax=ax4)
#
#         plt.tight_layout()
#         plt.show()
#
#         # Сохраняем график
#         plt.savefig('biometric_test_results.png', dpi=150, bbox_inches='tight')
#         print("\n📊 График сохранен как 'biometric_test_results.png'")
#
#     def generate_report(self, all_results):
#         """Генерирует подробный отчет в формате JSON"""
#         report = {
#             'timestamp': datetime.now().isoformat(),
#             'system_info': {
#                 'tolerance_default': self.system.tolerance,
#                 'data_dir': self.system.data_dir,
#                 'models_dir': self.system.models_dir
#             },
#             'test_config': {
#                 'train_dir': str(self.train_dir),
#                 'test_dir': str(self.test_dir)
#             },
#             'results': {}
#         }
#
#         for tolerance, results in all_results.items():
#             total = results['total']
#             if total > 0:
#                 accuracy = results['correct'] / total
#                 far = results['false_positive'] / (total * (total - 1)) if total > 1 else 0
#                 frr = results['false_negative'] / total
#
#                 report['results'][f'tolerance_{tolerance:.3f}'] = {
#                     'tolerance': tolerance,
#                     'total_tests': total,
#                     'correct': results['correct'],
#                     'false_positive': results['false_positive'],
#                     'false_negative': results['false_negative'],
#                     'accuracy': accuracy,
#                     'far': far,
#                     'frr': frr,
#                     'avg_similarity': np.mean([d['similarity'] for d in results['details']]) if results[
#                         'details'] else 0,
#                     'details': results['details']
#                 }
#
#         # Сохраняем отчет
#         report_file = f'biometric_test_report_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
#         with open(report_file, 'w', encoding='utf-8') as f:
#             json.dump(report, f, indent=2, ensure_ascii=False)
#
#         print(f"\n📄 Подробный отчет сохранен в: {report_file}")
#         return report
#
#
# def main():
#     """Главная функция тестирования"""
#     print("=" * 70)
#     print("    АВТОМАТИЗИРОВАННОЕ ТЕСТИРОВАНИЕ БИОМЕТРИЧЕСКОЙ СИСТЕМЫ")
#     print("=" * 70)
#
#     # Создаем экземпляр системы
#     # Если класс в другом файле, раскомментируйте следующую строку
#     # from face_biometric_system import FaceBiometricSystem
#     system = FaceBiometricSystem()
#
#     # Создаем тестер
#     tester = FaceBiometricTester(system, people_dir="Люди")
#
#     # Регистрируем всех пользователей
#     if not tester.register_all_users():
#         print("❌ Не удалось зарегистрировать пользователей. Проверьте папку 'Люди/1'")
#         return
#
#     # Тестируем с разными порогами
#     tolerance_values = [0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.10, 0.12]
#
#     print("\n" + "=" * 70)
#     print("ВЫБОР РЕЖИМА ТЕСТИРОВАНИЯ")
#     print("=" * 70)
#     print("1. Полное тестирование (все пороги)")
#     print("2. Быстрое тестирование (основные пороги)")
#     print("3. Настройка системы вручную")
#
#     choice = input("\nВаш выбор (1-3): ").strip()
#
#     if choice == '1':
#         results = tester.test_all_users(tolerance_values)
#         tester.plot_results(results)
#         tester.generate_report(results)
#
#     elif choice == '2':
#         results = tester.test_all_users([0.04, 0.05, 0.06, 0.07, 0.08])
#         tester.plot_results(results)
#         tester.generate_report(results)
#
#     elif choice == '3':
#         print("\n🔧 РУЧНАЯ НАСТРОЙКА")
#         print("-" * 50)
#
#         # Показываем текущие настройки
#         print(f"Текущий порог: {system.tolerance * 100:.1f}%")
#
#         # Тестируем с текущим порогом
#         results = tester.test_all_users([system.tolerance])
#
#         print("\n⚙️ РЕКОМЕНДАЦИИ ПО НАСТРОЙКЕ:")
#         if results and system.tolerance in results:
#             current_results = results[system.tolerance]
#             accuracy = current_results['correct'] / current_results['total']
#
#             if accuracy < 0.8:
#                 print("  • Низкая точность! Рекомендуется:")
#                 print("    1. Улучшить качество изображений")
#                 print("    2. Добавить выравнивание лиц")
#                 print("    3. Использовать более точные модели детекции")
#             elif accuracy < 0.95:
#                 print("  • Хорошая точность, но есть потенциал для улучшения")
#                 print("  • Попробуйте отрегулировать порог вниз или вверх на 1-2%")
#             else:
#                 print("  • Отличная точность! Система готова к использованию")
#
#     else:
#         print("❌ Неверный выбор")
#
#     print("\n" + "=" * 70)
#     print("ТЕСТИРОВАНИЕ ЗАВЕРШЕНО")
#     print("=" * 70)
#
#
# if __name__ == "__main__":
#     main()


import cv2
import numpy as np
import json
import os
import sys
from pathlib import Path
from datetime import datetime
import matplotlib.pyplot as plt
from collections import defaultdict

# Добавляем путь к родительской директории для импорта основного класса
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Импортируем ваш класс
from FaseBiometricSystem import FaceBiometricSystem


class FaceBiometricTester:
    """Класс для тестирования биометрической системы"""

    def __init__(self, system, people_dir="Люди"):
        """
        Инициализация тестера

        Args:
            system: экземпляр FaceBiometricSystem
            people_dir: директория с папками 1 и 2
        """
        self.system = system
        self.people_dir = Path(people_dir)
        self.train_dir = self.people_dir / "1"
        self.test_dir = self.people_dir / "2"

        # Результаты тестирования
        self.results = {}

    def get_people_list(self):
        """Получает список людей из папки 1"""
        people = []
        if self.train_dir.exists():
            for file in self.train_dir.iterdir():
                if file.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp']:
                    name = file.stem  # Имя без расширения
                    people.append(name)
        return people

    def get_test_images(self):
        """Получает список тестовых изображений из папки 2"""
        test_images = []
        if self.test_dir.exists():
            for file in self.test_dir.iterdir():
                if file.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp']:
                    test_images.append(file)
        return test_images

    def register_all_users(self):
        """Регистрирует всех пользователей из папки 1"""
        print("\n" + "=" * 70)
        print("РЕГИСТРАЦИЯ ПОЛЬЗОВАТЕЛЕЙ")
        print("=" * 70)

        people = self.get_people_list()

        if not people:
            print("❌ Ошибка: Не найдены изображения в папке 'Люди/1'")
            return False

        print(f"Найдено {len(people)} пользователей для регистрации:")
        for i, person in enumerate(people, 1):
            print(f"  {i}. {person}")

        successful = []
        failed = []

        for person in people:
            image_path = self.train_dir / f"{person}.jpg"
            # Проверяем другие расширения
            if not image_path.exists():
                for ext in ['.jpeg', '.png', '.bmp']:
                    alt_path = self.train_dir / f"{person}{ext}"
                    if alt_path.exists():
                        image_path = alt_path
                        break

            print(f"\n📝 Регистрация: {person}")
            if self.system.register_user(person, "file", str(image_path)):
                successful.append(person)
            else:
                failed.append(person)

        print("\n" + "=" * 70)
        print("РЕЗУЛЬТАТЫ РЕГИСТРАЦИИ")
        print("=" * 70)
        print(f"✅ Успешно: {len(successful)}/{len(people)}")
        if failed:
            print(f"❌ Неудачно: {failed}")

        return len(successful) > 0

    def test_all_users(self, tolerance_values=None):
        """
        Тестирует всех пользователей с разными порогами ошибки
        Считает ошибки первого и второго рода

        Args:
            tolerance_values: список значений порога для тестирования
        """
        if tolerance_values is None:
            tolerance_values = [0.03, 0.05, 0.07, 0.10, 0.12, 0.15]

        print("\n" + "=" * 70)
        print("ТЕСТИРОВАНИЕ СИСТЕМЫ")
        print("=" * 70)

        registered_users = self.get_people_list()
        test_images = self.get_test_images()

        if not registered_users:
            print("❌ Ошибка: Нет зарегистрированных пользователей")
            return

        if not test_images:
            print("❌ Ошибка: Нет тестовых изображений в папке 'Люди/2'")
            return

        print(f"Зарегистрировано пользователей: {len(registered_users)}")
        print(f"Тестовых изображений: {len(test_images)}")

        # Результаты для разных порогов
        all_results = {}

        for tolerance in tolerance_values:
            print(f"\n{'=' * 70}")
            print(f"📊 ТЕСТИРОВАНИЕ С ПОРОГОМ: {tolerance * 100:.1f}%")
            print(f"{'=' * 70}")

            self.system.tolerance = tolerance

            # Статистика
            genuine_attempts = 0  # Попытки "свой" vs "свой"
            impostor_attempts = 0  # Попытки "чужой" vs "свой"
            false_rejections = 0  # Ошибки 1 рода (FRR) - не пустили своего
            false_acceptances = 0  # Ошибки 2 рода (FAR) - пустили чужого

            details = []

            # Для каждой тестовой фотографии
            for test_image_path in test_images:
                test_person_name = test_image_path.stem  # Реальное имя человека на фото

                # Проверяем, есть ли этот человек в системе
                is_registered = test_person_name in registered_users

                # Пытаемся аутентифицироваться под каждым зарегистрированным пользователем
                for registered_user in registered_users:
                    # Определяем тип попытки
                    is_genuine = (test_person_name == registered_user)

                    if is_genuine:
                        genuine_attempts += 1
                    else:
                        impostor_attempts += 1

                    # Выполняем аутентификацию (с отключенным выводом и визуализацией)
                    auth_result = self._authenticate_silent(registered_user, str(test_image_path))

                    was_accepted = auth_result.get('authenticated', False)
                    similarity = auth_result.get('similarity', 0)
                    avg_error = auth_result.get('avg_error', 1)

                    # Фиксируем ошибки
                    if is_genuine and not was_accepted:
                        false_rejections += 1  # Ошибка 1 рода
                        error_type = "FRR"
                    elif not is_genuine and was_accepted:
                        false_acceptances += 1  # Ошибка 2 рода
                        error_type = "FAR"
                    else:
                        error_type = "OK"

                    # Сохраняем детали
                    details.append({
                        'test_person': test_person_name,
                        'attempted_as': registered_user,
                        'is_genuine': is_genuine,
                        'was_accepted': was_accepted,
                        'similarity': similarity,
                        'avg_error': avg_error,
                        'error_type': error_type
                    })

                    # Выводим прогресс
                    status = "✅" if (is_genuine and was_accepted) or (not is_genuine and not was_accepted) else "❌"
                    print(f"{status} {test_person_name} → {registered_user}: "
                          f"{'СВОЙ' if is_genuine else 'ЧУЖОЙ'} | "
                          f"{'ПУЩЕН' if was_accepted else 'ОТКЛОНЕН'} | "
                          f"схожесть: {similarity * 100:.1f}%")

            # Вычисляем метрики
            frr = false_rejections / genuine_attempts if genuine_attempts > 0 else 0
            far = false_acceptances / impostor_attempts if impostor_attempts > 0 else 0
            accuracy = 1 - (false_rejections + false_acceptances) / (genuine_attempts + impostor_attempts)

            # Сохраняем результаты
            results = {
                'tolerance': tolerance,
                'genuine_attempts': genuine_attempts,
                'impostor_attempts': impostor_attempts,
                'false_rejections': false_rejections,
                'false_acceptances': false_acceptances,
                'frr': frr,
                'far': far,
                'accuracy': accuracy,
                'details': details
            }

            all_results[tolerance] = results

            # Выводим сводку
            self._print_summary(results)

        self.results = all_results
        return all_results

    def _authenticate_silent(self, username, image_path):
        """
        Выполняет аутентификацию без вывода на экран и визуализации
        """
        # # Сохраняем оригинальный stdout
        # original_stdout = sys.stdout
        #
        # # Перенаправляем stdout в null с поддержкой UTF-8
        # sys.stdout = open(os.devnull, 'w', encoding='utf-8')
        #
        # # Перенаправляем stderr тоже
        # original_stderr = sys.stderr
        # sys.stderr = open(os.devnull, 'w', encoding='utf-8')

        # Сохраняем оригинальные настройки вывода
        original_stdout = sys.stdout
        sys.stdout = open(os.devnull, 'w')

        try:
            # Временно отключаем визуализацию
            original_show_result = self.system.authenticate_user.__defaults__

            # Выполняем аутентификацию
            result = self.system.authenticate_user(
                username,
                "file",
                image_path,
                show_resalt=False  # Отключаем показ результатов
            )

            # Восстанавливаем вывод
            sys.stdout.close()
            sys.stdout = original_stdout

            return result
        except Exception as e:
            # Восстанавливаем вывод
            sys.stdout.close()
            sys.stdout = original_stdout
            print(f"Ошибка при аутентификации {username}: {e}")
            return {'authenticated': False, 'similarity': 0, 'avg_error': 1}

    def _print_summary(self, results):
        """Выводит сводку результатов для одного порога"""
        print(f"\n{'─' * 70}")
        print(f"📊 СТАТИСТИКА ДЛЯ ПОРОГА {results['tolerance'] * 100:.1f}%")
        print(f"{'─' * 70}")
        print(f"Попытки 'свой-свой': {results['genuine_attempts']}")
        print(f"Попытки 'чужой-свой': {results['impostor_attempts']}")
        print(f"\n❌ Ошибки 1 рода (FRR - не пустили своего): {results['false_rejections']} "
              f"({results['frr'] * 100:.2f}%)")
        print(f"❌ Ошибки 2 рода (FAR - пустили чужого): {results['false_acceptances']} "
              f"({results['far'] * 100:.4f}%)")
        print(f"\n✅ Общая точность: {results['accuracy'] * 100:.2f}%")
        print(f"{'─' * 70}")

    def plot_results(self):
        """Визуализирует результаты тестирования"""
        if not self.results:
            print("❌ Нет результатов для визуализации")
            return

        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle('Анализ биометрической системы (ошибки 1 и 2 рода)', fontsize=16, fontweight='bold')

        # Подготовка данных
        tolerances = sorted(self.results.keys())
        frr_values = [self.results[t]['frr'] * 100 for t in tolerances]
        far_values = [self.results[t]['far'] * 100 for t in tolerances]
        accuracy_values = [self.results[t]['accuracy'] * 100 for t in tolerances]

        # 1. График FRR и FAR
        ax1 = axes[0, 0]
        ax1.plot(tolerances, frr_values, 'r^-', linewidth=2, markersize=8, label='FRR (Ошибка 1 рода)', color='red')
        ax1.plot(tolerances, far_values, 'bs-', linewidth=2, markersize=8, label='FAR (Ошибка 2 рода)', color='blue')
        ax1.set_xlabel('Порог ошибки (tolerance)', fontsize=12)
        ax1.set_ylabel('Ошибки, %', fontsize=12)
        ax1.set_title('Ошибки 1 и 2 рода', fontsize=12, fontweight='bold')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        ax1.set_xlim(min(tolerances) - 0.005, max(tolerances) + 0.005)

        # 2. Точность системы
        ax2 = axes[0, 1]
        ax2.plot(tolerances, accuracy_values, 'gD-', linewidth=2, markersize=8, color='green')
        ax2.set_xlabel('Порог ошибки (tolerance)', fontsize=12)
        ax2.set_ylabel('Точность, %', fontsize=12)
        ax2.set_title('Общая точность системы', fontsize=12, fontweight='bold')
        ax2.grid(True, alpha=0.3)
        ax2.set_ylim(0, 105)

        # Находим оптимальный порог
        best_idx = np.argmax(accuracy_values)
        best_tolerance = tolerances[best_idx]
        best_accuracy = accuracy_values[best_idx]
        ax2.plot(best_tolerance, best_accuracy, 'ro', markersize=12,
                 label=f'Оптимум: {best_tolerance * 100:.1f}% ({best_accuracy:.1f}%)')
        ax2.legend()

        # 3. Баланс FRR и FAR
        ax3 = axes[1, 0]
        # Находим точку пересечения
        diff = np.abs(np.array(frr_values) - np.array(far_values))
        cross_idx = np.argmin(diff)

        ax3.plot(frr_values, far_values, 'o-', linewidth=2, markersize=8)
        ax3.plot(frr_values[cross_idx], far_values[cross_idx], 'ro', markersize=12,
                 label=f'Баланс: порог {tolerances[cross_idx] * 100:.1f}%')
        ax3.set_xlabel('FRR (Ошибка 1 рода), %', fontsize=12)
        ax3.set_ylabel('FAR (Ошибка 2 рода), %', fontsize=12)
        ax3.set_title('Компромисс между ошибками (DET-кривая)', fontsize=12, fontweight='bold')
        ax3.grid(True, alpha=0.3)
        ax3.legend()

        # Добавляем аннотации для каждого порога
        for i, (frr, far, tol) in enumerate(zip(frr_values, far_values, tolerances)):
            ax3.annotate(f'{tol * 100:.0f}%', (frr, far),
                         textcoords="offset points", xytext=(5, 5), ha='center', fontsize=8)

        # 4. Количество ошибок
        ax4 = axes[1, 1]
        false_rejections = [self.results[t]['false_rejections'] for t in tolerances]
        false_acceptances = [self.results[t]['false_acceptances'] for t in tolerances]

        x = np.arange(len(tolerances))
        width = 0.35

        bars1 = ax4.bar(x - width / 2, false_rejections, width, label='FRR (не пустили своего)', color='red', alpha=0.7)
        bars2 = ax4.bar(x + width / 2, false_acceptances, width, label='FAR (пустили чужого)', color='blue', alpha=0.7)

        ax4.set_xlabel('Порог ошибки (tolerance)', fontsize=12)
        ax4.set_ylabel('Количество ошибок', fontsize=12)
        ax4.set_title('Абсолютное количество ошибок', fontsize=12, fontweight='bold')
        ax4.set_xticks(x)
        ax4.set_xticklabels([f'{t * 100:.0f}%' for t in tolerances])
        ax4.legend()
        ax4.grid(True, alpha=0.3, axis='y')

        # Добавляем значения на столбцы
        for bars in [bars1, bars2]:
            for bar in bars:
                height = bar.get_height()
                if height > 0:
                    ax4.annotate(f'{int(height)}',
                                 xy=(bar.get_x() + bar.get_width() / 2, height),
                                 xytext=(0, 3), textcoords="offset points",
                                 ha='center', va='bottom', fontsize=8)

        plt.tight_layout()
        plt.show()

        # Сохраняем график
        plt.savefig('biometric_test_results.png', dpi=150, bbox_inches='tight')
        print("\n📊 График сохранен как 'biometric_test_results.png'")

    def print_optimal_threshold(self):
        """Выводит рекомендации по оптимальному порогу"""
        if not self.results:
            print("❌ Нет результатов для анализа")
            return

        print("\n" + "=" * 70)
        print("🎯 ОПТИМИЗАЦИЯ ПОРОГА ОШИБКИ")
        print("=" * 70)

        # Находим порог с максимальной точностью
        best_accuracy = max(self.results.items(), key=lambda x: x[1]['accuracy'])

        # Находим порог с балансом FRR и FAR
        best_balance = min(self.results.items(),
                           key=lambda x: abs(x[1]['frr'] - x[1]['far']))

        # Находим порог с минимальными ошибками
        min_errors = min(self.results.items(),
                         key=lambda x: x[1]['false_rejections'] + x[1]['false_acceptances'])

        print(f"\n📌 По максимальной точности:")
        print(f"   • Порог: {best_accuracy[0] * 100:.1f}%")
        print(f"   • Точность: {best_accuracy[1]['accuracy'] * 100:.2f}%")
        print(f"   • FRR: {best_accuracy[1]['frr'] * 100:.2f}%")
        print(f"   • FAR: {best_accuracy[1]['far'] * 100:.4f}%")

        print(f"\n📌 По балансу FRR и FAR:")
        print(f"   • Порог: {best_balance[0] * 100:.1f}%")
        print(f"   • FRR: {best_balance[1]['frr'] * 100:.2f}%")
        print(f"   • FAR: {best_balance[1]['far'] * 100:.4f}%")
        print(f"   • Разница: {abs(best_balance[1]['frr'] - best_balance[1]['far']) * 100:.2f}%")

        print(f"\n📌 По минимальному количеству ошибок:")
        print(f"   • Порог: {min_errors[0] * 100:.1f}%")
        print(f"   • Всего ошибок: {min_errors[1]['false_rejections'] + min_errors[1]['false_acceptances']}")
        print(f"   • FRR: {min_errors[1]['frr'] * 100:.2f}%")
        print(f"   • FAR: {min_errors[1]['far'] * 100:.4f}%")

        print(f"\n💡 РЕКОМЕНДАЦИЯ:")
        if best_accuracy[1]['accuracy'] > 0.9:
            print(f"   ✅ Система показывает отличные результаты!")
            print(f"   🎯 Рекомендуемый порог: {best_accuracy[0] * 100:.1f}%")
        elif best_accuracy[1]['accuracy'] > 0.8:
            print(f"   👍 Система показывает хорошие результаты")
            print(f"   🎯 Рекомендуемый порог: {best_accuracy[0] * 100:.1f}%")
        else:
            print(f"   ⚠️ Точность системы низкая. Рекомендуется:")
            print(f"   1. Улучшить качество изображений")
            print(f"   2. Добавить предобработку (выравнивание, нормализацию)")
            print(f"   3. Использовать более точные модели детекции")

    def generate_report(self):
        """Генерирует подробный отчет в формате JSON"""
        if not self.results:
            print("❌ Нет результатов для генерации отчета")
            return

        report = {
            'timestamp': datetime.now().isoformat(),
            'system_info': {
                'tolerance_default': self.system.tolerance,
                'data_dir': self.system.data_dir,
                'models_dir': self.system.models_dir
            },
            'test_info': {
                'train_dir': str(self.train_dir),
                'test_dir': str(self.test_dir),
                'registered_users': self.get_people_list(),
                'test_images': [str(p) for p in self.get_test_images()]
            },
            'results': {}
        }

        for tolerance, results in self.results.items():
            report['results'][f'tolerance_{tolerance:.3f}'] = {
                'tolerance': tolerance,
                'genuine_attempts': results['genuine_attempts'],
                'impostor_attempts': results['impostor_attempts'],
                'false_rejections': results['false_rejections'],
                'false_acceptances': results['false_acceptances'],
                'frr_percent': results['frr'] * 100,
                'far_percent': results['far'] * 100,
                'accuracy_percent': results['accuracy'] * 100,
                'details': results['details']
            }

        # Сохраняем отчет
        report_file = f'biometric_test_report_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
        with open(report_file, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)

        print(f"\n📄 Подробный отчет сохранен в: {report_file}")
        return report


def main():
    """Главная функция тестирования"""
    print("=" * 70)
    print("    ТЕСТИРОВАНИЕ БИОМЕТРИЧЕСКОЙ СИСТЕМЫ")
    print("    Ошибки 1 рода (FRR) и 2 рода (FAR)")
    print("=" * 70)

    # Создаем экземпляр системы
    system = FaceBiometricSystem()

    # Создаем тестер
    tester = FaceBiometricTester(system, people_dir="Люди")

    # Регистрируем всех пользователей из папки "1"
    if not tester.register_all_users():
        print("❌ Не удалось зарегистрировать пользователей. Проверьте папку 'Люди/1'")
        return

    # Тестируем с разными порогами
    tolerance_values = [0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.10, 0.12]

    print("\n" + "=" * 70)
    print("НАЧАЛО ТЕСТИРОВАНИЯ")
    print("=" * 70)
    print(f"Будут протестированы пороги: {[f'{t * 100:.0f}%' for t in tolerance_values]}")

    input("\nНажмите Enter для начала тестирования...")

    # Запускаем тестирование
    results = tester.test_all_users(tolerance_values)

    # Визуализируем результаты
    tester.plot_results()

    # Выводим оптимальный порог
    tester.print_optimal_threshold()

    # Сохраняем отчет
    tester.generate_report()

    print("\n" + "=" * 70)
    print("ТЕСТИРОВАНИЕ ЗАВЕРШЕНО")
    print("=" * 70)


if __name__ == "__main__":
    main()