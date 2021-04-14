import os
import sys
sys.path.append(os.path.join(os.getcwd(), 'src'))

from config.constants import DOC_LABELS
from models.model_xray import ModelXray
from layers.pretrained import load_pretrained

class DocumentProcessor():

    def __init__(self):
        self.model = load_pretrained(model_class=ModelXray, model_dir='model', param_map=None)

    def predict(self, X, device=None):
        Y = self.model.predict(X=X, device=device)
        return [ y[DOC_LABELS] for y in Y]
