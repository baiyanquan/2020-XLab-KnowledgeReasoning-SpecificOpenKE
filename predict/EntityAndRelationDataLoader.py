import os


class EntityAndRelationDataLoader(object):
    def __init__(self, base_path, o_m_base_path):
        self.base_path = base_path
        self.o_m_base_path = o_m_base_path

    def acquire_entity(self):
        entity_list = []
        with open(self.base_path+'entity2id.txt') as f:
            f.readline()
            for line in f.readlines():
                entity_list.append(line.split('\t')[0])
        f.close()
        return entity_list

    def acquire_o_m_base_path(self):
        o_m_entity_list = []
        with open(self.o_m_base_path+'entity2id.txt') as f:
            f.readline()
            for line in f.readlines():
                o_m_entity_list.append(line.split('\t')[0])
        f.close()
        return o_m_entity_list

    def acquire_path(self):
        path_vec_list = []

        path_dict = {}
        with open(self.base_path+'pathList.txt') as f:
            f.readline()
            i = 0
            for line in f.readlines():
                performance_service = line.split('\t')[0]
                if performance_service not in path_dict.keys():
                    path_dict[performance_service] = [i]
                else:
                    path_dict[performance_service].append(i)
                i += 1
        f.close()
        return path_dict