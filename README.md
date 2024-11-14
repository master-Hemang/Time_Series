### テーマ: 自己教師あり学習を用いた時系列分類精度向上

#### 課題概要

本課題では、自己教師あり学習を用いて時系列データの分類精度を向上させることを目指します。自己教師あり学習とは、データのラベルがない状況で有用な表現を学習する手法であり、特に大規模な未ラベルデータを効果的に利用できる点が特徴です。今回は、UWaveGestureLibraryデータセットに対する時系列分類精度の向上を目的として、UWaveGestureLibrary以外のデータセットを用いて表現学習したモデルを用いて、ジェスチャー認識の分類精度を向上させることを目標とします。

#### 作業目安

- **全体期間**: 2週間
- **作業時間**: 24 ~ 40時間（1週間あたり12 ~ 20時間）
- **タイムライン(想定)**:
  - **1週目**: データセットの準備、論文サーベイ、自己教師あり学習の実装
  - **2週目**: 分類モデルの構築、性能検証、スライド作成

#### 課題内容

1. **データセットの準備**
   - **概要**: UWaveGestureLibraryデータセットを使用し、データのクリーニングと前処理を行います。このデータセットには、さまざまなジェスチャーに対応する時系列データが含まれています。
   - **URL**: [UWaveGestureLibraryデータセット](https://www.timeseriesclassification.com/description.php?Dataset=UWaveGestureLibrary)
   - **目標**: データの欠損値処理やスケーリングなどを行い、分析に適した形式に整えます。
   - **備考**: レポジトリに含まれるGesture.zipを使用してください。

2. **論文サーベイ**
   - **概要**: 自己教師あり学習に関する最新の研究論文を数本サーベイし、時系列データの分類における応用事例を調査します。
   - **目標**: 自己教師あり学習の技術的背景を理解し、時系列データへの適用方法を学びます。

3. **自己教師あり学習の実装**
   - **概要**: サーベイした手法の中から1つを選び、自己教師あり学習を実装します。選択した手法を用いて、時系列データの特徴表現を学習します。
   - **目標**: 未ラベルデータから有用な特徴を抽出し、分類モデルの精度を向上させます。
   - **備考**: 事前学習用のデータセットにはUWaveGestureLibrary以外のデータセットを使用してください。指定のデータセットはありませんがレポジトリに含まれるHAR.zipを使用しても問題ありません。

4. **分類モデルの構築**
   - **概要**: 学習された表現を使用して、時系列データの分類モデルを構築します。
   - **目標**: 自己教師あり学習で得られた特徴を活用し、精度の高い分類モデルを作成します。
   - **備考**: 構築するモデルに指定はありませんが、なぜそのモデルを選択したのか、その理由を説明できるようにはしてください。

5. **性能検証**
   - **概要**: 分類モデルの精度を評価し、自己教師あり学習を用いた場合の性能向上を確認します。従来の手法と比較して、分類精度の向上を定量的に評価します。
   - **目標**: 提案手法の有効性を定量的に検証し、改善点を明らかにします。
   - **備考**: 評価指標はご自身で選定してください。ただし、選定理由は説明できるようにはしてください。

6. **成果物の作成**
   - **概要**: 論文サーベイの結果、実装手順、性能検証の結果をまとめた資料を作成します。
   - **目標**: 実務で活用できるレベルの資料を作成し、技術内容をわかりやすく伝えるスキルを養います。

#### 提出物

1. **実装したプログラム**:
   - **提出方法**: 自身のGitHubアカウントにてリポジトリを作成し、コードをプッシュします。
   - **README**: 実行方法を記載し、プログラムの概要を説明します。

2. **プレゼンテーション用のスライド**:
   - **構成例**:
     - **背景**: なぜ自己教師あり学習を使ったアプローチをとる必要があったのか、どのような課題があったのか
     - **目的**: 検証の目的を明確にし、達成すべき目標を示します
     - **技術概要**: 用いた手法とその選択理由を説明
     - **評価指標**: 使用した評価指標の選定理由を含め、検証内容を説明
     - **検証内容と結果**: 検証結果とその考察、改善案
     - **まとめ**: 全体のまとめと今後の展望

3. **評価項目**
   - **技術力観点**:
     - 適切な評価指標の設計ができているか
     - 選択したアプローチに対する明確な理由があるか
     - コードの可読性（コードの構造化、コメントの適切さなど）
     - EDAやモデルの学習結果から適切な仮説が導かれ、検証が行われているか
     - 調査した文献の内容を理解できているか

   - **ビジネス観点**:
     - 技術的な内容をクライアントにわかりやすく説明できるか
     - 作成された資料全体のストーリーに一貫性があるか
     - 資料の各ページの内容（タイトル、ボディ、メッセージ）に一貫性があるか
     - プレゼンテーションでわかりやすく説明できているか
     - 決められた時間内にアウトプットを出せているか


### 補足資料

- **スライドライティングの参考資料**:
  - [スライドライティングの基本](https://note.com/powerpoint_jp/n/n812a673ce2ab)
  - [スライド作成のテクニック](https://note.com/powerpoint_jp/n/n9a8fd26ee181)
  - [Lecture on Slide Writing (Slideshare)](https://www.slideshare.net/slideshow/lecture-on-slide-writing/103255387)

- **自己教師あり学習の参考文献**:
  - [Self-supervised Learning (arXiv)](https://arxiv.org/abs/2301.05712)


### Theme: Improving time series classification accuracy using self-supervised learning

#### Project overview

In this project, we aim to improve the classification accuracy of time series data using self-supervised learning. Self-supervised learning is a method for learning useful expressions in situations where the data is unlabeled, and is characterized by its ability to effectively use large-scale unlabeled data. In this project, we aim to improve the classification accuracy of gesture recognition by using a model that has learned expressions using a dataset other than UWaveGestureLibrary, with the aim of improving the time series classification accuracy for the UWaveGestureLibrary dataset.

#### Estimated Work

- **Total duration**: 2 weeks
- **Work hours**: 24 ~ 40 hours (12 ~ 20 hours per week)
- **Timeline (expected)**:
- **Week 1**: Prepare the dataset, research papers, implement self-supervised learning
- **Week 2**: Build a classification model, verify performance, create slides

#### Assignment content

1. **Prepare the dataset**
- **Overview**: Use the UWaveGestureLibrary dataset to clean and preprocess the data. This dataset contains time series data corresponding to various gestures.
- **URL**: [UWaveGestureLibrary dataset](https://www.timeseriesclassification.com/description.php?Dataset=UWaveGestureLibrary)
- **Goal**: Handle missing values ​​and scale the data to make it suitable for analysis.
- **Notes**: Use Gesture.zip included in the repository.

2. **Paper Survey**

- **Overview**: Survey several of the latest research papers on self-supervised learning and investigate application cases in time series data classification.

- **Objective**: Understand the technical background of self-supervised learning and learn how to apply it to time series data.

3. **Implementation of self-supervised learning**

- **Overview**: Choose one of the surveyed methods and implement self-supervised learning. Use the selected method to learn feature representations of time series data.

- **Objective**: Extract useful features from unlabeled data and improve the accuracy of classification models.

- **Notes**: Use a dataset other than UWaveGestureLibrary for pre-training. There is no specified dataset, but you can use HAR.zip included in the repository.

4. **Building a classification model**

- **Overview**: Use the learned representation to build a classification model for time series data.
- **Objective**: Create a highly accurate classification model by utilizing the features obtained by self-supervised learning.

- **Note**: There is no specification for the model to be built, but please be able to explain why you selected that model.

5. **Performance verification**

- **Overview**: Evaluate the accuracy of the classification model and confirm the performance improvement when self-supervised learning is used. Quantitatively evaluate the improvement in classification accuracy compared to conventional methods.

- **Objective**: Quantitatively verify the effectiveness of the proposed method and clarify areas for improvement.

- **Note**: Please select your own evaluation indicators. However, please be able to explain the reason for your selection.

6. **Deliverable creation**

- **Overview**: Create a document summarizing the results of the paper survey, the implementation procedure, and the results of performance verification.

- **Objective**: Create a document at a level that can be used in practice and develop the skills to communicate technical content in an easy-to-understand manner.

#### Submission

1. **Implemented program**:
- **How ​​to submit**: Create a repository in your own GitHub account and push the code.
- **README**: Describe how to run the program and provide an overview of the program.

2. **Presentation slides**:

- **Example of configuration**:

- **Background**: Why was it necessary to adopt an approach using self-supervised learning? What challenges did you face?

- **Objective**: Clarify the purpose of the verification and indicate the goals to be achieved.

- **Technical overview**: Explain the method used and the reason for its selection.

- **Evaluation indicators**: Explain the verification content, including the reason for selecting the evaluation indicators used.

- **Verification content and results**: Verification results and their considerations and improvement proposals.

- **Summary**: Overall summary and future outlook.

3. **Evaluation items**

- **Technical ability perspective**:

- Are appropriate evaluation indicators designed?

- Are there clear reasons for the selected approach?

- Code readability (code structuring, appropriate comments, etc.)

- Are appropriate hypotheses derived from the EDA and model learning results and verified?

- Do you understand the contents of the literature surveyed?

- **Business perspective**:

- Can you explain the technical content to clients in an easy-to-understand way?

- Is there consistency in the overall story of the created materials?

Is the content of each page of the document consistent (title, body, message)?

- Is the presentation easy to understand?

- Can you produce output within the allotted time?

### Supplementary materials

- **Reference materials for slide writing**:

- [Basics of slide writing](https://note.com/powerpoint_jp/n/n812a673ce2ab)

- [Techniques for creating slides](https://note.com/powerpoint_jp/n/n9a8fd26ee181)

- [Lecture on Slide Writing (Slideshare)](https://www.slideshare.net/slideshow/lecture-on-slide-writing/103255387)

- **References for self-supervised learning**:

- [Self-supervised Learning (arXiv)](https://arxiv.org/abs/2301.05712)