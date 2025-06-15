import os
import glob
import torch
import numpy as np
from PIL import Image
from torchvision import transforms
from scipy import ndimage
import csv
from datetime import datetime
import matplotlib.pyplot as plt

# Configuração de paths relativos
MAFCN_DIR = 'C:\\Projects\\Processamento trab final\\MAFCN\\pretrained_model\\vgg16bn-encoder.pkl'
METERS_PER_PIXEL = 0.25 # Assumindo 0.25m² por pixel

from MAFCN.model.MA_FCN1 import NetFB
from MAFCN.model.Unet import Unet
from MAFCN.utils.tools import untransform_pred

class BuildingChangeDetector:
    def __init__(self, model_type='MAFCN'):
        self.model_type = model_type
        self.model = self._load_model()
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

    def _load_model(self):
        """Carrega o modelo pré-treinado"""
        if self.model_type == 'MAFCN':
            model = NetFB(class_num=2)
            encoder_path = MAFCN_DIR
            model.features.load_state_dict(torch.load(encoder_path))
        else:
            model = Unet(num_classes=2)
            fcn_path = os.path.join(MAFCN_DIR, 'pretrained_model', 'fcn8s_from_caffe.pth')
            fcn_weights = torch.load(fcn_path)
            model.load_state_dict({k: v for k, v in fcn_weights.items() if k in model.state_dict()})

        model.eval()
        model.cuda()
        return model

    def _load_image(self, image_path_base):
        """Carrega imagens com tratamento de erros"""
        for ext in ['.png', '.tif', '.jpg', '.jpeg']:
            try:
                img_path = image_path_base + ext
                if os.path.exists(img_path):
                    img = Image.open(img_path).convert('RGB')
                    return img, img_path
            except Exception as e:
                print(f"Erro ao carregar {img_path}: {str(e)}")
        return None, None

    def _calculate_metrics(self, pred_mask, output):
        """Calcula métricas adicionais da predição"""
        binary_mask = np.array(pred_mask).astype(bool)

        # Contagem de prédios (componentes conectados)
        labeled, num_buildings = ndimage.label(binary_mask)

        # Área total em pixels (assumindo 0.25m² por pixel para imagens aéreas)
        area = binary_mask.sum() * METERS_PER_PIXEL

        # Confiança média da predição
        confidence = torch.sigmoid(output).mean().item()

        # Tamanho médio dos prédios
        building_sizes = [np.sum(labeled == i) for i in range(1, num_buildings+1)]
        avg_size = np.mean(building_sizes) if building_sizes else 0

        return {
            'count': num_buildings,
            'area': area,
            'confidence': confidence,
            'avg_size': avg_size,
            'max_size': max(building_sizes) if building_sizes else 0,
            'min_size': min(building_sizes) if building_sizes else 0
        }

    def predict(self, image_path_base):
        """Executa a predição para uma imagem"""
        img, img_path = self._load_image(image_path_base)
        if img is None:
            return None

        # Pré-processamento
        img_tensor = self.transform(img).unsqueeze(0).cuda()

        # Predição
        with torch.no_grad():
            if self.model_type == 'MAFCN':
                output, _, _, _, _ = self.model(img_tensor)
            else:
                output = self.model(img_tensor)

        # Pós-processamento
        pred_mask = untransform_pred(output[0]).convert('1')
        metrics = self._calculate_metrics(pred_mask, output[0])

        return {
            'metrics': metrics,
            'pred_mask': pred_mask,
            'image_path': img_path,
            'image_size': img.size
        }

    def classify_damage(self, before, after):
        """Classifica o tipo de dano baseado nas métricas"""
        change = after['count'] - before['count']
        area_change = after['area'] - before['area']

        if before['count'] == 0:
            return "new_development"

        percent_change = (change / before['count']) * 100

        if percent_change < -40:
            return "severe_destruction"
        elif -40 <= percent_change < -10:
            return "partial_destruction"
        elif -10 <= percent_change < 0:
            return "minor_damage"
        elif percent_change == 0:
            if abs(area_change) > before['area'] * 0.1:  # Mudança de área sem mudança na contagem
                return "structural_changes"
            return "intact"
        else:
            return "new_construction"

    def generate_report(self, before_folder, after_folder, output_dir='results'):
        """Gera relatório completo de comparação"""
        # Criar diretório de saída
        os.makedirs(output_dir, exist_ok=True)

        # Encontrar imagens correspondentes
        extensions = ['*.png', '*.tif', '*.jpg', '*.jpeg']
        before_images = {}
        after_images = {}

        for ext in extensions:
            for path in glob.glob(os.path.join(before_folder, ext)):
                base_name = os.path.splitext(os.path.basename(path))[0]
                before_images[base_name] = path

            for path in glob.glob(os.path.join(after_folder, ext)):
                base_name = os.path.splitext(os.path.basename(path))[0]
                after_images[base_name] = path

        common_names = set(before_images.keys()) & set(after_images.keys())

        if not common_names:
            print("Nenhuma imagem correspondente encontrada!")
            return

        # Processar cada par de imagens
        results = []
        for name in common_names:
            print(f"\nProcessando: {name}")

            # Predições
            before_pred = self.predict(os.path.join(before_folder, name))
            after_pred = self.predict(os.path.join(after_folder, name))

            if not before_pred or not after_pred:
                continue

            # Classificação de danos
            damage_class = self.classify_damage(before_pred['metrics'], after_pred['metrics'])

            # Resultados
            result = {
                'filename': name,
                'before_count': before_pred['metrics']['count'],
                'after_count': after_pred['metrics']['count'],
                'count_change': after_pred['metrics']['count'] - before_pred['metrics']['count'],
                'before_area': before_pred['metrics']['area'],
                'after_area': after_pred['metrics']['area'],
                'area_change': after_pred['metrics']['area'] - before_pred['metrics']['area'],
                'before_confidence': before_pred['metrics']['confidence'],
                'after_confidence': after_pred['metrics']['confidence'],
                'damage_class': damage_class,
                'analysis_date': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                'before_image': before_pred['image_path'],
                'after_image': after_pred['image_path']
            }

            # Visualização (opcional)
            self._save_comparison_plot(
                before_pred, after_pred, result,
                os.path.join(output_dir, f"{name}_comparison.png"))

            results.append(result)

            # Log detalhado
            self._print_detailed_report(result)

        # Salvar CSV
        csv_path = os.path.join(output_dir, 'building_changes_report.csv')
        self._save_to_csv(results, csv_path)

        # Gerar resumo
        self._generate_summary(results, output_dir)

        return results

    def _save_comparison_plot(self, before, after, result, output_path):
        """Gera visualização comparativa"""
        plt.figure(figsize=(15, 10))

        # Imagem antes
        plt.subplot(2, 2, 1)
        plt.imshow(Image.open(before['image_path']))
        plt.title(f"Antes\nPrédios: {result['before_count']}\nÁrea: {result['before_area']:.2f}m²")

        # Máscara antes
        plt.subplot(2, 2, 2)
        plt.imshow(before['pred_mask'], cmap='gray')
        plt.title("Detecção antes")

        # Imagem depois
        plt.subplot(2, 2, 3)
        plt.imshow(Image.open(after['image_path']))
        plt.title(f"Depois\nPrédios: {result['after_count']}\nÁrea: {result['after_area']:.2f}m²")

        # Máscara depois
        plt.subplot(2, 2, 4)
        plt.imshow(after['pred_mask'], cmap='gray')
        plt.title("Detecção depois")

        plt.suptitle(
            f"Análise: {result['filename']}\n"
            f"Mudança: {result['count_change']} prédios | "
            f"Área: {result['area_change']:.2f}m²\n"
            f"Classificação: {result['damage_class'].replace('_', ' ').title()}",
            y=1.02
        )

        plt.tight_layout()
        plt.savefig(output_path, bbox_inches='tight', dpi=150)
        plt.close()

    def _print_detailed_report(self, result):
        """Exibe relatório detalhado no console"""
        print(f"""
        {'='*80}
        ANÁLISE DE MUDANÇAS: {result['filename']}
        {'-'*80}
        ANTES:
        - Prédios detectados: {result['before_count']}
        - Área construída: {result['before_area']:.2f} m²
        - Confiança média: {result['before_confidence']*100:.1f}%
        
        DEPOIS:
        - Prédios detectados: {result['after_count']}
        - Área construída: {result['after_area']:.2f} m²
        - Confiança média: {result['after_confidence']*100:.1f}%
        
        MUDANÇAS:
        - Variação no número de prédios: {result['count_change']} ({result['count_change']/result['before_count']*100:.1f}%)
        - Variação na área construída: {result['area_change']:.2f} m²
        - Classificação de danos: {result['damage_class'].replace('_', ' ').title()}
        
        Data da análise: {result['analysis_date']}
        {'='*80}
        """)

    def _save_to_csv(self, results, output_path):
        """Salva resultados em CSV"""
        fieldnames = [
            'filename', 'before_count', 'after_count', 'count_change',
            'before_area', 'after_area', 'area_change',
            'before_confidence', 'after_confidence',
            'damage_class', 'analysis_date', 'before_image', 'after_image'
        ]

        with open(output_path, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(results)

        print(f"\nRelatório salvo em: {output_path}")

    def _generate_summary(self, results, output_dir):
        """Gera um resumo estatístico"""
        if not results:
            return

        summary = {
            'total_images': len(results),
            'total_buildings_before': sum(r['before_count'] for r in results),
            'total_buildings_after': sum(r['after_count'] for r in results),
            'total_area_before': sum(r['before_area'] for r in results),
            'total_area_after': sum(r['after_area'] for r in results),
            'damage_distribution': {},
            'most_affected': max(results, key=lambda x: abs(x['count_change']))
        }

        # Distribuição de danos
        for result in results:
            damage_class = result['damage_class']
            summary['damage_distribution'][damage_class] = summary['damage_distribution'].get(damage_class, 0) + 1

        # Salvar resumo
        summary_path = os.path.join(output_dir, 'summary_report.txt')
        with open(summary_path, 'w', encoding='utf-8') as f:
            f.write("RELATÓRIO RESUMO DE DETECÇÃO DE MUDANÇAS\n")
            f.write("="*50 + "\n\n")
            f.write(f"Imagens analisadas: {summary['total_images']}\n")
            f.write(f"Total de prédios antes: {summary['total_buildings_before']}\n")
            f.write(f"Total de prédios depois: {summary['total_buildings_after']}\n")
            f.write(f"Variação total: {summary['total_buildings_after'] - summary['total_buildings_before']}\n")
            f.write(f"Área construída antes: {summary['total_area_before']:.2f} m²\n")
            f.write(f"Área construída depois: {summary['total_area_after']:.2f} m²\n")
            f.write(f"Variação de área: {summary['total_area_after'] - summary['total_area_before']:.2f} m²\n\n")

            f.write("DISTRIBUIÇÃO DE DANOS:\n")
            for damage, count in summary['damage_distribution'].items():
                f.write(f"- {damage.replace('_', ' ').title()}: {count} áreas ({count/summary['total_images']*100:.1f}%)\n")

            f.write("\nÁREA MAIS AFETADA:\n")
            f.write(f"- Nome: {summary['most_affected']['filename']}\n")
            f.write(f"- Mudança: {summary['most_affected']['count_change']} prédios\n")
            f.write(f"- Classificação: {summary['most_affected']['damage_class'].replace('_', ' ').title()}\n")

        print(f"Resumo salvo em: {summary_path}")

if __name__ == '__main__':
    # Configuração
    BEFORE_FOLDER = './past'
    AFTER_FOLDER = './present'
    OUTPUT_DIR = './results'

    # Inicializar detector
    detector = BuildingChangeDetector(model_type='MAFCN')

    # Executar análise
    results = detector.generate_report(
        before_folder=BEFORE_FOLDER,
        after_folder=AFTER_FOLDER,
        output_dir=OUTPUT_DIR
    )