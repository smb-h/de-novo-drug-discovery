# De Novo Drug Design

This project explores innovative approaches in drug discovery using recurrent and deep neural networks. The goal is to generate novel drug formulas with specialized structures, demonstrating the potential of artificial intelligence to expedite the identification of promising compounds for pharmaceutical development.

## Getting Started

### Installation

   ```bash
   pip install -r requirements.txt
   ```

### Preprocessing

   ```bash
   python -m drug_design.data.preprocessing_molinf
   ```

### Training the Model

   ```bash
   python -m drug_design.models.train
   ```

### Checking Accuracy & Loss

   ```bash
   tensorboard --logdir reports/<date>/<experiment_name>/logs
   ```

### Prediction

   ```bash
   python -m drug_design.models.predict
   ```

### Fine-tuning the Model

   ```bash
   python -m drug_design.models.fine_tune
   ```

## Contribution

- SMBH

## License

This project is licensed under the MIT. See the LICENSE file for details.


