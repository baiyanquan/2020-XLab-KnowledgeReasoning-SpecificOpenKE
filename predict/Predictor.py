from .ModelLoader import ModelLoader
from .EntityAndRelationDataLoader import EntityAndRelationDataLoader


class Predictor(object):
    def __init__(self):
        self.entity = []

    def data_load(self, model_base_path='./embedding/', entity_relation_base_path = './benchmarks/FKB/'):
        self.model = ModelLoader(model_base_path)
        self.entityAndRelation = EntityAndRelationDataLoader(entity_relation_base_path)
        self.entity = self.entityAndRelation.acquire_entity()
        self.path = self.entityAndRelation.acquire_path()

    def predict(self, target_performance):
        performance_service = target_performance.split(':')[0]
