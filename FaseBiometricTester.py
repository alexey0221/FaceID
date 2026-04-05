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


# Импортируем ваш класс (предполагается, что он в отдельном файле)
# Если класс в этом же файле, то импорт не нужен
# from FaceBiometricSystem import FaceBiometricSystem
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
        self.results = {
            'authentications': [],
            'statistics': {},
            'errors': [],
            'confusion_matrix': defaultdict(lambda: defaultdict(int))
        }

    def get_people_list(self):
        """Получает список людей из папки 1"""
        people = []
        if self.train_dir.exists():
            for file in self.train_dir.iterdir():
                if file.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp']:
                    name = file.stem  # Имя без расширения
                    people.append(name)
        return people

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
            if not os.path.exists(image_path):#.exists():
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

        Args:
            tolerance_values: список значений порога для тестирования
        """
        if tolerance_values is None:
            tolerance_values = [0.03, 0.05, 0.07, 0.10, 0.12, 0.15]

        print("\n" + "=" * 70)
        print("ТЕСТИРОВАНИЕ СИСТЕМЫ")
        print("=" * 70)

        people = self.get_people_list()

        if not people:
            print("❌ Ошибка: Нет зарегистрированных пользователей")
            return

        # Результаты для разных порогов
        all_results = {}

        for tolerance in tolerance_values:
            print(f"\n📊 Тестирование с порогом ошибки: {tolerance * 100:.1f}%")
            print("-" * 50)

            self.system.tolerance = tolerance
            results = self.run_tests(people)
            all_results[tolerance] = results

            self.print_summary(results)

        # Анализ и оптимизация
        self.analyze_and_optimize(all_results, people)

        return all_results

    def run_tests(self, people):
        """Запускает тесты для всех пользователей"""
        results = {
            'total': 0,
            'correct': 0,
            'false_positive': 0,
            'false_negative': 0,
            'details': [],
            'scores': {}
        }

        for person in people:
            # Находим тестовое изображение
            test_image = self.test_dir / f"{person}.jpg"
            if not test_image.exists():
                for ext in ['.jpeg', '.png', '.bmp']:
                    alt_path = self.test_dir / f"{person}{ext}"
                    if alt_path.exists():
                        test_image = alt_path
                        break

            if not test_image.exists():
                print(f"⚠️ Предупреждение: Нет тестового изображения для {person}")
                continue

            print(f"\n🔐 Тестирование: {person}")

            # Пытаемся аутентифицироваться как этот пользователь
            auth_result = self.system.authenticate_user(person, "file", str(test_image))

            is_correct = auth_result.get('authenticated', False)
            similarity = auth_result.get('similarity', 0)
            avg_error = auth_result.get('avg_error', 1)

            result_detail = {
                'person': person,
                'authenticated': is_correct,
                'similarity': similarity,
                'avg_error': avg_error,
                'is_correct': is_correct  # Для правильного пользователя
            }

            results['details'].append(result_detail)
            results['total'] += 1

            if is_correct:
                results['correct'] += 1
                results['scores'][person] = similarity
                print(f"  ✅ Верно (схожесть: {similarity * 100:.1f}%)")
            else:
                results['false_negative'] += 1
                print(f"  ❌ Ошибка FN (схожесть: {similarity * 100:.1f}%)")

            # Проверяем cross-аутентификацию (false positives)
            for other_person in people:
                if other_person != person:
                    print(f"  🔄 Cross-test: {person} как {other_person}")
                    cross_result = self.system.authenticate_user(other_person, "file", str(test_image))

                    if cross_result.get('authenticated', False):
                        results['false_positive'] += 1
                        print(f"    ⚠️ Ложное срабатывание: {person} распознан как {other_person}")
                        self.results['confusion_matrix'][person][other_person] += 1

        return results

    def print_summary(self, results):
        """Выводит сводку результатов"""
        print("\n" + "=" * 50)
        print("СВОДКА ТЕСТИРОВАНИЯ")
        print("=" * 50)

        accuracy = results['correct'] / results['total'] if results['total'] > 0 else 0
        far = results['false_positive'] / (results['total'] * (results['total'] - 1)) if results['total'] > 1 else 0
        frr = results['false_negative'] / results['total'] if results['total'] > 0 else 0

        print(f"📊 Общая статистика:")
        print(f"  • Всего тестов: {results['total']}")
        print(f"  • Верных: {results['correct']}")
        print(f"  • Ложных отказов (FN): {results['false_negative']}")
        print(f"  • Ложных срабатываний (FP): {results['false_positive']}")
        print(f"\n📈 Метрики:")
        print(f"  • Точность (Accuracy): {accuracy * 100:.2f}%")
        print(f"  • FAR (False Acceptance Rate): {far * 100:.4f}%")
        print(f"  • FRR (False Rejection Rate): {frr * 100:.2f}%")

        if results['scores']:
            avg_similarity = np.mean(list(results['scores'].values()))
            print(f"  • Средняя схожесть: {avg_similarity * 100:.2f}%")

    def analyze_and_optimize(self, all_results, people):
        """
        Анализирует результаты и предлагает оптимальные параметры

        Args:
            all_results: результаты для разных порогов
            people: список пользователей
        """
        print("\n" + "=" * 70)
        print("АНАЛИЗ И ОПТИМИЗАЦИЯ")
        print("=" * 70)

        # Находим оптимальный порог
        best_tolerance = None
        best_accuracy = 0
        best_metrics = None

        for tolerance, results in all_results.items():
            accuracy = results['correct'] / results['total'] if results['total'] > 0 else 0

            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_tolerance = tolerance
                best_metrics = results
        if best_tolerance:
            print(f"\n🎯 Оптимальный порог ошибки: {best_tolerance * 100:.1f}%")
        if best_accuracy:
            print(f"   Достигнутая точность: {best_accuracy * 100:.2f}%")

        # Анализ ошибок
        print(f"\n📋 Детальный анализ ошибок:")

        if best_metrics:
            # Анализ false negatives
            fn_cases = [d for d in best_metrics['details'] if not d['is_correct']]
            if fn_cases:
                print(f"\n  ❌ Ложные отказы (FN):")
                for case in fn_cases:
                    print(f"    - {case['person']}: схожесть {case['similarity'] * 100:.1f}%, "
                          f"ошибка {case['avg_error'] * 100:.1f}%")

            # Анализ false positives из confusion matrix
            if self.results['confusion_matrix']:
                print(f"\n  ⚠️ Ложные срабатывания (FP):")
                for real, fakes in self.results['confusion_matrix'].items():
                    for fake, count in fakes.items():
                        if count > 0:
                            print(f"    - {real} распознан как {fake} ({count} раз)")

        # Рекомендации по улучшению
        print(f"\n💡 РЕКОМЕНДАЦИИ ПО УЛУЧШЕНИЮ:")

        if best_accuracy < 0.8:
            print(f"  1. 🔧 Текущая точность низкая ({best_accuracy * 100:.1f}%)")
            print(f"     • Увеличьте количество точек лица (улучшите детекцию)")
            print(f"     • Используйте более качественные изображения для регистрации")
            print(f"     • Добавьте предобработку изображений (выравнивание, нормализацию)")

        # Анализ вариативности
        if best_metrics and best_metrics['scores']:
            similarities = list(best_metrics['scores'].values())
            variance = np.var(similarities)
            if variance > 0.05:
                print(f"  2. 📊 Высокая вариативность схожести ({variance:.4f})")
                print(f"     • Некоторые пользователи распознаются хуже других")
                print(f"     • Рассмотрите индивидуальные пороги для сложных случаев")

        # Рекомендации по порогу
        print(f"  3. 🎛️ Настройка порога:")
        if best_tolerance:
            print(f"     • Текущий оптимальный: {best_tolerance * 100:.1f}%")

        # Проверяем, нужно ли уменьшить FAR
        if best_metrics:
            far = best_metrics['false_positive'] / (best_metrics['total'] * (best_metrics['total'] - 1)) if \
            best_metrics['total'] > 1 else 0
            if far > 0.01:  # Больше 1%
                print(f"     • Высокий FAR ({far * 100:.2f}%), уменьшите порог до {best_tolerance * 100 - 1:.1f}%")
            else:
                print(f"     • Хороший FAR ({far * 100:.2f}%)")

    def plot_results(self, all_results):
        """Визуализирует результаты тестирования"""
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        fig.suptitle('Анализ биометрической системы', fontsize=16)

        # 1. График точности от порога
        ax1 = axes[0, 0]
        tolerances = list(all_results.keys())
        accuracies = [r['correct'] / r['total'] if r['total'] > 0 else 0 for r in all_results.values()]

        ax1.plot(tolerances, accuracies, 'bo-', linewidth=2, markersize=8)
        ax1.set_xlabel('Порог ошибки', fontsize=12)
        ax1.set_ylabel('Точность', fontsize=12)
        ax1.set_title('Зависимость точности от порога', fontsize=12)
        ax1.grid(True, alpha=0.3)
        ax1.set_xlim(min(tolerances) - 0.01, max(tolerances) + 0.01)
        ax1.set_ylim(0, 1.05)

        # 2. FAR и FRR
        ax2 = axes[0, 1]
        far_values = []
        frr_values = []

        for tolerance, results in all_results.items():
            total = results['total']
            far = results['false_positive'] / (total * (total - 1)) if total > 1 else 0
            frr = results['false_negative'] / total if total > 0 else 0
            far_values.append(far)
            frr_values.append(frr)

        ax2.plot(tolerances, far_values, 'r^-', linewidth=2, markersize=8, label='FAR')
        ax2.plot(tolerances, frr_values, 'gs-', linewidth=2, markersize=8, label='FRR')
        ax2.set_xlabel('Порог ошибки', fontsize=12)
        ax2.set_ylabel('Ошибки', fontsize=12)
        ax2.set_title('FAR и FRR', fontsize=12)
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        # 3. Распределение схожести
        ax3 = axes[1, 0]
        if all_results:
            best_tolerance = max(all_results.keys(), key=lambda t: all_results[t]['correct'] / all_results[t]['total'])
            similarities = [d['similarity'] for d in all_results[best_tolerance]['details']]

            ax3.hist(similarities, bins=10, alpha=0.7, color='blue', edgecolor='black')
            ax3.axvline(x=best_tolerance, color='red', linestyle='--',
                        label=f'Порог: {best_tolerance * 100:.1f}%')
            ax3.set_xlabel('Схожесть', fontsize=12)
            ax3.set_ylabel('Частота', fontsize=12)
            ax3.set_title('Распределение схожести', fontsize=12)
            ax3.legend()
            ax3.grid(True, alpha=0.3)

        # 4. Confusion matrix heatmap (упрощенная)
        ax4 = axes[1, 1]
        if self.results['confusion_matrix']:
            people = list(self.results['confusion_matrix'].keys())
            matrix = np.zeros((len(people), len(people)))

            for i, real in enumerate(people):
                for j, fake in enumerate(people):
                    if i != j:
                        matrix[i, j] = self.results['confusion_matrix'][real].get(fake, 0)

            im = ax4.imshow(matrix, cmap='YlOrRd', interpolation='nearest')
            ax4.set_xticks(range(len(people)))
            ax4.set_yticks(range(len(people)))
            ax4.set_xticklabels(people, rotation=45, ha='right', fontsize=8)
            ax4.set_yticklabels(people, fontsize=8)
            ax4.set_xlabel('Распознан как', fontsize=12)
            ax4.set_ylabel('Реальный пользователь', fontsize=12)
            ax4.set_title('Матрица ошибок', fontsize=12)
            plt.colorbar(im, ax=ax4)

        plt.tight_layout()
        plt.show()

        # Сохраняем график
        plt.savefig('biometric_test_results.png', dpi=150, bbox_inches='tight')
        print("\n📊 График сохранен как 'biometric_test_results.png'")

    def generate_report(self, all_results):
        """Генерирует подробный отчет в формате JSON"""
        report = {
            'timestamp': datetime.now().isoformat(),
            'system_info': {
                'tolerance_default': self.system.tolerance,
                'data_dir': self.system.data_dir,
                'models_dir': self.system.models_dir
            },
            'test_config': {
                'train_dir': str(self.train_dir),
                'test_dir': str(self.test_dir)
            },
            'results': {}
        }

        for tolerance, results in all_results.items():
            total = results['total']
            if total > 0:
                accuracy = results['correct'] / total
                far = results['false_positive'] / (total * (total - 1)) if total > 1 else 0
                frr = results['false_negative'] / total

                report['results'][f'tolerance_{tolerance:.3f}'] = {
                    'tolerance': tolerance,
                    'total_tests': total,
                    'correct': results['correct'],
                    'false_positive': results['false_positive'],
                    'false_negative': results['false_negative'],
                    'accuracy': accuracy,
                    'far': far,
                    'frr': frr,
                    'avg_similarity': np.mean([d['similarity'] for d in results['details']]) if results[
                        'details'] else 0,
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
    print("    АВТОМАТИЗИРОВАННОЕ ТЕСТИРОВАНИЕ БИОМЕТРИЧЕСКОЙ СИСТЕМЫ")
    print("=" * 70)

    # Создаем экземпляр системы
    # Если класс в другом файле, раскомментируйте следующую строку
    # from face_biometric_system import FaceBiometricSystem
    system = FaceBiometricSystem()

    # Создаем тестер
    tester = FaceBiometricTester(system, people_dir="Люди")

    # Регистрируем всех пользователей
    if not tester.register_all_users():
        print("❌ Не удалось зарегистрировать пользователей. Проверьте папку 'Люди/1'")
        return

    # Тестируем с разными порогами
    tolerance_values = [0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.10, 0.12]

    print("\n" + "=" * 70)
    print("ВЫБОР РЕЖИМА ТЕСТИРОВАНИЯ")
    print("=" * 70)
    print("1. Полное тестирование (все пороги)")
    print("2. Быстрое тестирование (основные пороги)")
    print("3. Настройка системы вручную")

    choice = input("\nВаш выбор (1-3): ").strip()

    if choice == '1':
        results = tester.test_all_users(tolerance_values)
        tester.plot_results(results)
        tester.generate_report(results)

    elif choice == '2':
        results = tester.test_all_users([0.04, 0.05, 0.06, 0.07, 0.08])
        tester.plot_results(results)
        tester.generate_report(results)

    elif choice == '3':
        print("\n🔧 РУЧНАЯ НАСТРОЙКА")
        print("-" * 50)

        # Показываем текущие настройки
        print(f"Текущий порог: {system.tolerance * 100:.1f}%")

        # Тестируем с текущим порогом
        results = tester.test_all_users([system.tolerance])

        print("\n⚙️ РЕКОМЕНДАЦИИ ПО НАСТРОЙКЕ:")
        if results and system.tolerance in results:
            current_results = results[system.tolerance]
            accuracy = current_results['correct'] / current_results['total']

            if accuracy < 0.8:
                print("  • Низкая точность! Рекомендуется:")
                print("    1. Улучшить качество изображений")
                print("    2. Добавить выравнивание лиц")
                print("    3. Использовать более точные модели детекции")
            elif accuracy < 0.95:
                print("  • Хорошая точность, но есть потенциал для улучшения")
                print("  • Попробуйте отрегулировать порог вниз или вверх на 1-2%")
            else:
                print("  • Отличная точность! Система готова к использованию")

    else:
        print("❌ Неверный выбор")

    print("\n" + "=" * 70)
    print("ТЕСТИРОВАНИЕ ЗАВЕРШЕНО")
    print("=" * 70)


if __name__ == "__main__":
    main()