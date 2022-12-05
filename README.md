De Novo Drug Design
==============================

1. Install requirements
   
    $ pip install -r requirements.txt
    
2. Preprocessing
   
    $ python -m drug_design.data.preprocessing_molinf

3. Train model
   
    $ python -m drug_design.models.train

4. Check accuracy & loss
   
    $ tensorboard --logdir reports/<date>/<experiment_name>/logs

5. Predict
   
    $ python -m drug_design.models.predict

6. Fine-tune model
   
    $ python -m drug_design.models.fine_tune
