import ast
import json

from typing import Counter
import pandas as pd
from collections import Counter, defaultdict, namedtuple

from PolymerSolarCellsML.dataset.dataset_base_classes import PropertyValuePair
from PolymerSolarCellsML.dataset.property_extraction import PropertyExtractor

from PolymerSolarCellsML.dataset.constants import PC61BM, PC71BM, ICBA, C60

from rdkit import Chem
import re
from PolymerSolarCellsML.property_prediction import canonicalize

import logging

logger = logging.getLogger()
logging.basicConfig()
logger.setLevel(logging.INFO)

class DatasetParser():
    def __init__(self, property_name) -> None:
        # location of file and load data using pandas
        file = '../../dataset/polymer_solar_cell_curated_data.xlsx'
        self.df = pd.read_excel(file, sheet_name="Sheet1", keep_default_na=False)
        self.property_name = property_name
        # Load normalization dataset
        normalization_file = '../../metadata/normalized_polymer_dictionary.json'

        self.property_extractor = PropertyExtractor()

        with open(normalization_file, 'r') as fi:
            train_data_text = fi.read()

        self.polymer_metadata = json.loads(train_data_text)

        self.fullerene_60 = ['PCBM', 'PC_{61}BM', 'PC_{60}BM',  'PC61BM', '[6,6]-phenyl-C61-butyric acid methyl ester', \
             '[6,6]-phenyl C_{61} butyric acid methyl ester', '[6,6]-phenyl-C_{61}-butyricacid methyl ester', '[6,6]-phenyl-C_{61}-buytyric acid methyl ester', \
             'PC60BM', '[6,6]-phenyl-C61-butyric acid methyl ester', '[6,6]-phenyl-C_{61}-butyric acid methyl ester', \
             'PC_{61}BH', '[6,6]-phenyl C61-butyric acid methyl ester', 'phenyl-C_{61}-butyric acid methyl ester', 
             '[6,6]-phenyl C_{61}-butyric acid methyl ester', '[6,6]-phenylC_{61}-butyric acid methyl ester', '[6,6]-phenyl-C_{61} butyric acid methyl ester', '[60]PCBM', 
             '1-(3-methoxycarbonyl)-propyl-1-phenyl-(6,6)C61', '6,6-phenyl-C61-butyric acid methyl ester', '[6,6]-phenyl-C61 butyric acid methyl ester', 'phenyl-C61-butyric acid methyl ester', \
             '[6,6]-phenyl C61 butyric acid methyl ester', '[6,6]-phenyl C61-butyric acid methylester', '[6, 6]-phenyl-C_{61}-butyric acid methyl ester', '6.6-phenyl C_{61}-butyric acid methyl ester', '[6,6]-phenyl-C61-butyric-acidmethyl-ester'] # hexyl included here for simplicity
        self.fullerene_70 = ['PC_{70}BM', 'PC_{71}BM', 'PCBM[70]', 'PC70BM', 'PC71BM', '[6,6]-phenyl-C71-butyric acid methyl ester', \
                             '[6,6]-phenyl C_{71} butyric acid methyl ester', '[6,6]-phenyl-C_{71}-butyricacid methyl ester', '[6,6]-phenyl C71-butyric acid methyl ester', \
                             '[6,6]-phenyl-C71-butyric acid methylester', '[6,6]-phenyl-C_{71}-butyric acid methyl-ester', '(6,6)-phenyl-C71-butyric acid methyl ester', \
                             '[6,6]-phenyl-C_{71}-butyric acid methyl ester', '6,6-phenyl-C_{71}-butyric acid', '[70]PCBM', 'C70-PCBM', '{6,6}-phenyl-C71 butyric acid methyl ester', \
                             '[6,6]-phenyl C_{71}-butyric acid methyl ester', '[6,6]-phenyl-C71 butyric acid methyl ester', '(6,6)-phenyl C71 butyric acid methyl ester', \
                             '[6,6]-phenyl C71 butyric acid methyl ester', '[6,6]-phenyl- C71-butyric acid methyl ester', '(6,6)-phenyl C_{71}-butyric acid methyl ester']
        self.icba = ['ICBA', 'IPC_{60}BM', 'indene-PC_{60}BM', 'indene-C_{60}', 'Indene-C60', 'IC_{60}BA', "1′,1′′,4′,4′′-Tetrahydrodi[1,4]methanonaphthaleno[1,2:2′,3′,56,60:2′′,3′′][5,6]fullerene-C60", 'IC_{70}BA', 'indene-C_{70}', 'indene-C70', 'C4-BFCBA']
        # IC60BA and IC70BA are being clubbed for convenience as only two reported cases of IC70BA are in our dataset
        self.C60 = ['C_{60}','C60']
        
    def normalize_name(self, material):
        return next(
            (
                key
                for key, value in self.polymer_metadata.items()
                if material in value['coreferents']
                or material.lower() in value["coreferents"]
                or material.title() in value['coreferents']
                or material.upper() in value["coreferents"]
            ),
            material,
        )
        
    def normalize_fullerene_name(self, row):
        material = row["acceptor"]
        material = material.strip()
        if material.startswith("='") and material.endswith("'"):
            material = material[2:-1]
        if self.check_membership(material, self.fullerene_60):
            return 'PC61BM', PC61BM
        elif self.check_membership(material, self.fullerene_70):
            return 'PC71BM', PC71BM
        elif self.check_membership(material, self.icba):
            return 'ICBA', ICBA
        elif self.check_membership(material, self.C60):
            return 'C60', C60
        else:
            return material, row['acceptor_smiles']
    
    def check_membership(self, material, material_list):
        return (
            material in material_list
            or material.lower() in material_list
            or material.title() in material_list
            or material.upper() in material_list
        )
    

    def construct_NFA_normalization_dict(self):
        """Construct a dictionary of NFA names to correspond to their coreferents"""
        filtered_df = self.df[(self.df["fullerene_acceptor"]==0) & (self.df["curated"]==1)]
        self.NFA_normalization_dict = defaultdict(set)
        for index, row in filtered_df.iterrows():
            doi = row["DOI"]
            material = self.cleanup_material_name(row["acceptor"], doi)
            if row["acceptor_coreferents"]:
                material_coreferents = ast.literal_eval(row["acceptor_coreferents"])
                material_coreferents = [self.cleanup_material_name(item, doi) for item in material_coreferents]
            else:
                material_coreferents = [material]
            
            for item in self.NFA_normalization_dict:
                if any(mat_item in material_coreferents or mat_item.lower() in material_coreferents or mat_item.title() in material_coreferents or mat_item.upper() in material_coreferents for mat_item in self.NFA_normalization_dict[item]):
                    for material in material_coreferents:
                        self.NFA_normalization_dict[item].add(material)
                    break
            else:
                for mat in material_coreferents: self.NFA_normalization_dict[material].add(mat)
    
    def normalize_NFA_name(self, material, doi):
        """Normalize NFA names to their coreferents"""
        # First use regular normalization, if that does not work, then use NFA normalization
        material = self.cleanup_material_name(material, doi)
        for key, value in self.polymer_metadata.items():
            if material in value['coreferents'] or material.lower() in value["coreferents"] or material.title() in value['coreferents'] or material.upper() in value["coreferents"]:
                return key

        return next(
            (
                key
                for key, value in self.NFA_normalization_dict.items()
                if material in value
                or material.lower() in value
                or material.title() in value
                or material.upper() in value
            ),
            material,
        )

    
    def cleanup_material_name(self, material, doi):
        # Accounts for donor and acceptor names not mentioned in abstracts but mentoned in the body of the paper
        material = material.strip()
        if material.startswith("='") and material.endswith("'"): material = material[2:-1]
        # Undergo a normalization step here
        generic_identifiers = [f'P{integer}' for integer in range(10)]+['P', 'polymer', 'poly', 'RP', 'PP', 'poly(1)', 'poly(2)', 'poly(3)', 'poly(4)', 'poly(5)', 'P-1a', 'P-1b', 'P-1c', 'P_{1}', 'P_{2}', 'P_{3}']+[f'RP{integer}' for integer in range(10)]+[f'{integer}' for integer in range(10)]+[f'{integer}{char}' for integer in range(5) for char in 'abcdefg']+ [f'P{char}' for char in 'abcd']+[chr(item) for item in range(65, 91)] # What other generic identifiers are possible?
        if material in generic_identifiers:
            material = f'{material}_{doi}'
        else:
            material = self.normalize_name(material)
        
        return material
    
    def data_statistics(self):
        # Check how much data there is on donor and what data there is on PCE and other corresponding points
        # How many unique donor names and how many unique donor smile strings
        top_k = 25
        donor_names = list(self.df["donor"])
        donor_smiles = list(self.df["donor_smiles"])
        acceptor_names = list(self.df["acceptor"])
        acceptor_smiles = list(self.df["acceptor_smiles"])
        doi_list = list(self.df["DOI"])
        polymer_acceptor_smiles = set()
        organic_acceptor_smiles = set()
        donor_polymer_acceptor_pairs = defaultdict(int)
        donor_organic_acceptor_pairs = defaultdict(int)
        repeated_pairs = defaultdict(dict)
        ml_dataset = defaultdict(lambda: {'DOI': [], self.property_name: []})
        donor_dict = dict()
        acceptor_dict = dict()
        data_item = namedtuple('data_item', ['donor', 'donor_smiles', 'acceptor', 'acceptor_smiles', self.property_name.replace(' ', '_'), 'DOI'])

        for smile_string in acceptor_smiles:
            if smile_string:
                if '[*]' in smile_string:
                    polymer_acceptor_smiles.add(smile_string)
                else:
                    organic_acceptor_smiles.add(smile_string)
        for index, (donor_smile, acceptor_smile) in enumerate(zip(donor_smiles, acceptor_smiles)):
            if self.df["curated"][index]==1:
                if self.df["fullerene_acceptor"][index]==1:
                    if self.df["acceptor"][index] in self.fullerene_60:
                        acceptor_smile = PC61BM
                        self.df["acceptor"][index] = 'PC61BM'
                    elif self.df["acceptor"][index] in self.fullerene_70:
                        acceptor_smile = PC71BM
                        self.df["acceptor"][index] = 'PC71BM'
                    elif self.df["acceptor"][index] in self.icba:
                        acceptor_smile = ICBA
                        self.df["acceptor"][index] = 'ICBA'
                    elif self.df["acceptor"][index] in self.C60:
                        acceptor_smile = C60
                        self.df["acceptor"][index] = 'C60'

                if donor_smile or acceptor_smile:
                    donor = str(self.df["donor"][index])
                    acceptor = str(self.df["acceptor"][index])
                    if donor and donor==donor and len(donor)>1:
                        donor = self.cleanup_material_name(donor, index)
                    if acceptor and acceptor==acceptor and len(acceptor)>1:
                        acceptor = self.cleanup_material_name(acceptor, index)
                    if donor and donor==donor and acceptor==acceptor:
                        if donor != acceptor:
                            if donor_smile and donor not in donor_dict: donor_dict[donor] = donor_smile
                            if acceptor and acceptor_smile and acceptor not in acceptor_dict: acceptor_dict[acceptor] = acceptor_smile

                            if self.df[self.property_name][index]:
                                if self.df["property_metadata"][index]:
                                    prop_metadata = json.loads(self.df["property_metadata"][index])[self.property_name]
                                if not self.df['curated'][index]:
                                    prop_value = prop_metadata["property_numeric_value"]
                                elif self.df['curated'][index] and type(self.df[self.property_name][index])==str and prop_metadata: # Curated but PCE unchanged
                                    prop_value = prop_metadata["property_numeric_value"]
                                elif self.df['curated'][index] and type(self.df[self.property_name][index])==float:
                                    prop_value = self.df[self.property_name][index]*100 # Use the curated value entered
                                elif self.df['curated'][index] and type(self.df[self.property_name][index])==str and not prop_metadata:
                                    property_entity = PropertyValuePair(property_value=self.df[self.property_name][index])
                                    self.property_extractor.single_property_entity_postprocessing(property_entity)
                                    prop_value = property_entity.property_numeric_value
                                if prop_value<20:
                                    ml_dataset[(donor, donor_smile, acceptor, acceptor_smile)][self.property_name].append(prop_value)
                                    ml_dataset[(donor, donor_smile, acceptor, acceptor_smile)]["DOI"].append(self.df["DOI"][index])
                            
                            if '[*]' in acceptor_smile and donor_smile:
                                donor_polymer_acceptor_pairs[(donor, acceptor)] += 1
                            elif acceptor_smile and donor_smile:
                                donor_organic_acceptor_pairs[(donor, acceptor)] += 1
                        else:
                            if 'DOI' not in repeated_pairs[donor].keys():
                                repeated_pairs[donor]['DOI'] = []
                                repeated_pairs[donor]['count'] = 0
                            repeated_pairs[donor]['DOI'].append(self.df["DOI"][index])
                            repeated_pairs[donor]['count']+=1


        donor_names_counter = Counter(donor_names)
        donor_names_counter.pop('')
        donor_names_counter_top = dict(sorted(donor_names_counter.items(), key = lambda x: x[1], reverse=True)[:top_k])
        acceptor_names_counter = Counter(acceptor_names)
        acceptor_names_counter.pop('')
        polymer_acceptor_pairs_top = dict(sorted(donor_polymer_acceptor_pairs.items(), key = lambda x: x[1], reverse=True)[:top_k])
        organic_acceptor_pairs_top = dict(sorted(donor_organic_acceptor_pairs.items(), key = lambda x: x[1], reverse=True)[:top_k])
        acceptor_names_counter_top = dict(sorted(acceptor_names_counter.items(), key = lambda x: x[1], reverse=True)[:top_k])
        repeated_pairs_top = dict(sorted(repeated_pairs.items(), key = lambda x: x[1]['count'], reverse=True)[:top_k])

        # How many acceptors are polymers and how many are organics

        # How many pairs of donor acceptors and num points associated with the top few

        # report statistics
        print(f'Number of documents = {len(Counter(doi_list))}')
        print(f'Number of unique donor names = {len(donor_names_counter.keys())}')
        print(f'Top unique donor names = {donor_names_counter_top}')
        print(f'Number of unique donor smile string = {len(Counter(donor_smiles).keys())}')
        print(f'Number of unique acceptor names = {len(acceptor_names_counter.keys())}')
        print(f'Number of polymer acceptor smile strings = {len(polymer_acceptor_smiles)}')
        print(f'Number of organic acceptor smile strings = {len(organic_acceptor_smiles)}')
        print(f'Top unique acceptor names = {acceptor_names_counter_top}')
        print(f'Number of unique acceptor smiles = {len(Counter(acceptor_smiles).keys())}')
        print(f'Number of donor-organic acceptor pairs = {len(donor_organic_acceptor_pairs.keys())}')
        print(f'Number of donor-polymer acceptor pairs = {len(donor_polymer_acceptor_pairs.keys())}')
        print(f'Top unique donor polymer acceptor pairs = {polymer_acceptor_pairs_top}')
        print(f'Top unique donor organic acceptor pairs = {organic_acceptor_pairs_top}')
        print(f'Top repeated pairs = {repeated_pairs_top}')
        final_ml_dataset = [data_item(*key, value[self.property_name], value["DOI"]) for key, value in ml_dataset.items()]

        return donor_organic_acceptor_pairs, donor_polymer_acceptor_pairs, final_ml_dataset, donor_dict, acceptor_dict
    
    def auxiliary_data(self):
        """Print additional information about the dataset to put in thesis"""
        doi_all = set()
        donor_fullerene_acceptor_pairs = set()
        donors = set()
        donor_NFA_pairs = set()
        fullerene_acceptors = set()
        NFA_acceptors = set()
        donor_fullerene_datapoints = 0
        NFA_datapoints = 0
        doi_curated = set()
        curated_fullerene_acceptor_pairs = set()
        curated_NFA_pairs = set()
        curated_fullerene_datapoints = 0
        curated_NFA_datapoints = 0
        max_pce = 0
        donor_dict = dict()
        acceptor_dict = dict()
        ml_dataset = defaultdict(lambda: {'DOI': [], 'donor_acceptor_pair': [], self.property_name: []})
        final_ml_dataset = defaultdict(lambda: {'DOI': [], self.property_name: []})
        final_ml_dataset = []
        data_item = namedtuple('data_item', ['donor', 'donor_smiles', 'acceptor', 'acceptor_smiles', self.property_name.replace(' ', '_'), 'DOI'])
        self.construct_NFA_normalization_dict()
        nlp_extracted_data_file = '../../dataset/polymer_solar_cell_extracted_data.csv'
        nlp_extracted_data = pd.read_csv(nlp_extracted_data_file)

        for index, row in self.df.iterrows():
            # Get cumulative data for all points
            doi = row["DOI"]
            doi_all.add(doi)
            if row["donor"]!=row["donor"] or row["acceptor"]!=row["acceptor"]: # Skip nan values
                continue
            donor_normalized_name = self.cleanup_material_name(str(row["donor"]), doi)
            donors.add(donor_normalized_name)
            if row["fullerene_acceptor"]==1:
                acceptor_normalized_name, fullerene_smiles = self.normalize_fullerene_name(row)
                fullerene_acceptors.add(acceptor_normalized_name)
                donor_fullerene_acceptor_pairs.add((donor_normalized_name, acceptor_normalized_name))
                donor_fullerene_datapoints+=1
            else:
                acceptor_normalized_name = self.normalize_NFA_name(str(row["acceptor"]), doi)
                if not acceptor_normalized_name:
                    pass

                NFA_acceptors.add(acceptor_normalized_name)
                donor_NFA_pairs.add((donor_normalized_name, acceptor_normalized_name))
                NFA_datapoints+=1

            if row["curated"]==1:
                # get all data for curated points
                doi_curated.add(row["DOI"])
                if row["donor_smiles"]!='':
                    canonical_donor_smiles = canonicalize_smiles(row["donor_smiles"])
                else:
                    logger.info(f'Canonical donor smiles string is empty for DOI {doi}')
                    
                if row["fullerene_acceptor"]==1:
                    # Normalize fullerene acceptor names
                    canonical_acceptor_smiles = canonicalize_smiles(fullerene_smiles)
                    if (canonical_donor_smiles, canonical_acceptor_smiles) in curated_NFA_pairs:
                        logger.info(f'For {doi} curated fullerene pair {donor_normalized_name, acceptor_normalized_name} already exists in curated NFA pairs')
                    curated_fullerene_acceptor_pairs.add((canonical_donor_smiles, canonical_acceptor_smiles))
                    curated_fullerene_datapoints+=1
                else:
                    canonical_acceptor_smiles = canonicalize_smiles(row["acceptor_smiles"])
                    if (canonical_donor_smiles, canonical_acceptor_smiles) in curated_fullerene_acceptor_pairs:
                        logger.info(f'For {doi} curated NFA pair {donor_normalized_name, acceptor_normalized_name} already exists in curated fullerene pairs')
                    curated_NFA_pairs.add((canonical_donor_smiles, canonical_acceptor_smiles))
                    curated_NFA_datapoints+=1
                
                if not canonical_acceptor_smiles:
                    logger.info(f'Canonical acceptor smiles string is empty for DOI {doi}')
                
                if row[self.property_name]:
                    prop_value = self.property_value(row)

                    if prop_value>max_pce:
                        max_pce = prop_value
                        max_pair = (donor_normalized_name, acceptor_normalized_name)
                        max_doi = doi

                    ml_dataset[(canonical_donor_smiles, canonical_acceptor_smiles)]["DOI"].append(doi)
                    ml_dataset[(canonical_donor_smiles, canonical_acceptor_smiles)]["donor_acceptor_pair"].append((donor_normalized_name, acceptor_normalized_name))
                    ml_dataset[(canonical_donor_smiles, canonical_acceptor_smiles)][self.property_name].append(prop_value)
        
        donor_acceptor = dict()
        for (donor_smiles, acceptor_smiles), value in ml_dataset.items():
            donor_acceptor_pairs = Counter(value["donor_acceptor_pair"])
            donor, acceptor = donor_acceptor_pairs.most_common(1)[0][0]
            if not donor or not acceptor:
                logger.info(f'For DOI {value["DOI"]} donor or acceptor is empty')
            if (donor, acceptor) in donor_acceptor:
                logger.info(f'For DOI {value["DOI"]} donor acceptor pair {donor, acceptor} already exists in donor acceptor dictionary with DOI {donor_acceptor[(donor, acceptor)]["DOI"]}')
            year_list = nlp_extracted_data[nlp_extracted_data["DOI"].isin(value["DOI"])]["year"].tolist()
            if (donor_smiles, acceptor_smiles) in curated_fullerene_acceptor_pairs:
                fullerene_acceptor_status = True
            else:
                fullerene_acceptor_status = False
            donor_acceptor[(donor, acceptor)] = {self.property_name: value[self.property_name],
                                                 'DOI': value["DOI"],
                                                 'donor_smiles': donor_smiles,
                                                 'acceptor_smiles': acceptor_smiles,
                                                 'year': year_list,
                                                 'fullerene_acceptor': fullerene_acceptor_status
                                                 }
            donor_dict[donor] = donor_smiles
            acceptor_dict[acceptor] = acceptor_smiles
            final_ml_dataset.append(data_item(donor,donor_smiles, acceptor, acceptor_smiles, value[self.property_name], value["DOI"]))
        
        donors.remove('')
        NFA_acceptors.remove('')

        print('Statistics for complete dataset')
        print(f'Number of unique DOI = {len(doi_all)}')

        print(f'Number of estimated unique fullerene pairs = {len(donor_fullerene_acceptor_pairs)}')
        print(f'Number of estimated unique NFA pairs = {len(donor_NFA_pairs)}')

        print(f'Number of estimated unique donors = {len(donors)}')
        print(f'Number of estimated unique fullerene acceptors = {len(fullerene_acceptors)}')
        print(f'Number of estimated unique NFA acceptors = {len(NFA_acceptors)}')

        print(f'Number of estimated fullerene datapoints = {donor_fullerene_datapoints}')
        print(f'Number of estimated NFA datapoints = {NFA_datapoints}')

        print('Statistics for curated dataset')
        print(f'Number of unique DOI = {len(doi_curated)}')

        print(f'Number of curated fullerene pairs = {len(curated_fullerene_acceptor_pairs)}')
        print(f'Number of curated NFA pairs = {len(curated_NFA_pairs)}')

        print(f'Number of curated donors = {len(donor_dict.keys())}')
        print(f'Number of curated acceptors = {len(acceptor_dict.keys())}')

        print(f'Number of curated fullerene datapoints = {curated_fullerene_datapoints}')
        print(f'Number of curated NFA datapoints = {curated_NFA_datapoints}')

        print(f'Maximum PCE value = {max_pce} with system {max_pair} and DOI {max_doi}')

        return final_ml_dataset, donor_dict, acceptor_dict, donor_acceptor
    
    def property_value(self, row):
        """Parse the property value from the row"""
        # Assumes that % value is being parsed

        if type(row[self.property_name])==str: # Curated but PCE unchanged
            property_entity = PropertyValuePair(property_value=row[self.property_name])
            self.property_extractor.single_property_entity_postprocessing(property_entity)
            prop_value = property_entity.property_numeric_value
        
        elif type(row[self.property_name])==float and row[self.property_name]<1:
            prop_value = row[self.property_name]*100 # Use the curated value entered
        
        elif type(row[self.property_name])==float and row[self.property_name]>1:
            prop_value = row[self.property_name] # Use the curated value entered

        return prop_value

def canonicalize_smiles(smiles_string):
    smiles_string = smiles_string.strip()
    regexp = re.compile(r'\{(.*?)\}')
    try:
        if '[*]' in smiles_string: # Polymer canonical SMILES string
            if re.search(regexp, smiles_string):
                smiles_canonical_list = []
                for sub_smile in regexp.findall(smiles_string)[0].split(','):
                    smiles_canonical_list.append(canonicalize.canonicalize(sub_smile))
                canonical_smiles_string = '{'+','.join(smiles_canonical_list)+'}'
            else:
                canonical_smiles_string = canonicalize.canonicalize(smiles_string)
        else:
            canonical_smiles_string = Chem.CanonSmiles(smiles_string)
    except:
        print(f'{smiles_string} gave an error during canonicalization, using uncanonicalized SMILES string instead')
        canonical_smiles_string = smiles_string
    
    return canonical_smiles_string                