
# Proof-Of-Concept
## Nawigacja

- [POC Chexpert](resnet_chexpert.ipynb)
- [POC ResNet50](resnet_gradcam.ipynb)

## Opis

Proof-Of-Concept (PoC) został zaimplementowany w języku Python z wykorzystaniem biblioteki PyTorch. Kod źródłowy jest dostępny jako załącznik do raportu i może być uruchomiony na dowolnej maszynie z zainstalowanym środowiskiem Python i Jupyter.

Prototyp pokazuje, w jaki sposób można wykorzystać mechanizm Gradient CAM (Class Activation Mapping) do wizualizacji aktywacji w sieciach konwolucyjnych.  
W ramach prototypu zrealizowano:

1. Przygotowanie zbioru danych  
2. Dostosowanie modelu ResNet50 do klasyfikacji obrazów RTG klatki piersiowej  
3. Wizualizacja aktywacji w sieci  

Ze względu na ograniczone możliwości obliczeniowe zaimplementowany model nie jest precyzyjny i nie nadaje się do zastosowań produkcyjnych; jego głównym celem jest pokazanie możliwości wizualizacji aktywacji w sieciach konwolucyjnych.

## Zbiór danych

Do treningu klasyfikatora wykorzystano zbiór danych oparty na *CheXpert*.  
Wykorzystany zbiór można pobrać z [Kaggle](https://www.kaggle.com/datasets/ashery/chexpert).  
Zbiór zawiera 224 316 obrazów RTG klatki piersiowej z 65 240 pacjentów, z których każdy został opisany przez 14 różnych etykiet.

## Model podstawowy

Do klasyfikacji obrazów RTG klatki piersiowej wykorzystano zmodyfikowany model ResNet50.  
ResNet jest jednym z najpopularniejszych modeli w dziedzinie klasyfikacji obrazów i jest często wykorzystywany w zastosowaniach medycznych. Ponadto ResNet jest dostępny w wielu różnych implementacjach w bibliotekach takich jak PyTorch czy TensorFlow.

## Dostosowanie modelu

Model ResNet50 został dostosowany do klasyfikacji obrazów RTG klatki piersiowej poprzez zmianę warstwy wyjściowej.  
ResNet jest modelem przeznaczonym do klasyfikacji obrazów z bazy ImageNet, która zawiera obrazy z 1000 różnych klas.  
W związku z tym, ostatnia w pełni połączona warstwa modelu ResNet50 została zastąpiona warstwą z 14 neuronami, odpowiadającymi 14 etykietom z CheXpert.

Następnie model był trenowany na zbiorze danych CheXpert przez 1 epokę z wykorzystaniem algorytmu *AdamW*.  
Jako funkcję straty wykorzystano *BCEWithLogitsLoss*, a jako metrykę oceny jakości klasyfikacji *AUROC* (Area Under Receiver Operating Characteristic).

Z powodu ograniczonych możliwości obliczeniowych nie podjęto próby usprawnienia modelu poprzez hiperparametryzację czy zastosowanie bardziej zaawansowanych technik regularyzacji.  
Dostosowany model osiągnął jakość klasyfikacji na poziomie 0.5 AUROC, co jest wynikiem losowym i nie nadaje się do zastosowań produkcyjnych.

## Wizualizacja aktywacji

Do wizualizacji aktywacji wykorzystano mechanizm Gradient CAM (Class Activation Mapping).  
Pierwszym krokiem działania mechanizmu jest wybór ostatniej warstwy konwolucyjnej w modelu, która zawiera informacje o lokalizacji cech w obrazie.  
Następnie obliczane są gradienty aktywacji w tej warstwie względem klasy docelowej, co pozwala na określenie, które obszary obrazu były najbardziej istotne dla klasyfikacji.  
Obliczone gradienty są następnie agregowane w celu uzyskania mapy aktywacji, która jest nakładana na obraz wejściowy.

Do realizacji wizualizacji aktywacji wykorzystano biblioteki PyTorch, PIL.Image oraz pytorch_gradcam.  
W ramach prototypu możliwa jest wizualizacja aktywacji dla dowolnego obrazu RTG klatki piersiowej w obrębie wszystkich 14 klas dostępnych w zbiorze danych CheXpert.

Ponieważ prototyp jest oparty o model o niskiej jakości, wizualizacje aktywacji nie są wiarygodne i nie nadają się do zastosowań diagnostycznych.  
Efektywnie, prototyp pokazuje jedynie w jaki sposób można zaimplementować mechanizm GradCAM w sieciach konwolucyjnych oraz jakiego rodzaju wyników należy oczekiwać i przetwarzać w warstwie prezentacji.