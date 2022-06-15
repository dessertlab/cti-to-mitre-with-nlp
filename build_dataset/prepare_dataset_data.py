from stix2 import MemoryStore
from stix2 import FileSystemSource
from stix2 import Filter
from stix2 import parse

from nltk.tokenize import sent_tokenize
import pandas as pd
import re


#Takes a list of text and combines them into one large chunk of text.
def combine_text(list_of_text):
    combined_text = ' '.join(list_of_text)
    return combined_text

def repl(matchobj):
    return matchobj.group(1) + ","

def cleaning_data(text): 
	#text = text.lower()							#CountVectorize already puts all text in lower case by default
	text = re.sub('\(i.e.', '', text)
	text = re.sub('\[(.*?)\]', repl, text)
	text = re.sub('\(.*?\)', '', text)
	text = re.sub('\)', '', text)
	text = re.sub('\<\/?code\>', '', text)
	text = text.strip()
	return text

def remove_empty_lines(text):
	lines = text.split("\n")
	non_empty_lines = [line for line in lines if line.strip() != ""]

	string_without_empty_lines = ""
	for line in non_empty_lines:
		if line != "\n": 
			string_without_empty_lines += line + "\n"

	return string_without_empty_lines 

def map_subtec_to_tec(id):
	return id.split(".", 2)[0]

def add_description_from_capec(capec_id):
	capec_fs = FileSystemSource('../cti/capec/2.0')
	capec_filters = [Filter('type', '=', 'attack-pattern'), Filter("description", "!=", ""), Filter("external_references.external_id", "=", capec_id)]
	capec_attack_patterns = capec_fs.query(capec_filters)
	descriptions = []
	for ap in capec_attack_patterns:
		ap_parsed = parse(ap, allow_custom=True)
		descriptions.append(ap_parsed.description)
	return descriptions

#accessing ATT&CK by tecniques "x_mitre_deprecated"
def get_deprecated_list(src, parsed_data):
	deprecated_id_list = []
	deprecated_filter = [Filter('type', '=', 'attack-pattern'), Filter("x_mitre_deprecated", "=", True)]
	deprecated = src.query(deprecated_filter)
	for ap in deprecated:
		parsed_data = parse(ap, allow_custom=True)
		deprecated_id_list.append(parsed_data.id)
	# print("Deprecated techniques: " + str(len(deprecated_id_list)) + "\n")
	return deprecated_id_list

def get_attack_dict(src, attack_patterns):
	
	attack_ids_dict = {}																		# Dictionary of all tecniques
	attack_ids_names_dict = {}

	for ap in attack_patterns:
		ap_parsed = parse(ap, allow_custom= True)
		deprecated_list = get_deprecated_list(src, ap_parsed)									# Get the list of DEPRECATED attack patterns 
		for ref in ap_parsed.external_references:
			if ref.source_name == "mitre-attack" and ap_parsed.id not in deprecated_list:		# Consider only non-deprecated patterns
				attack_ids_dict[ap_parsed.id] = ref.external_id
				attack_ids_names_dict[ref.external_id] = ap_parsed.name
	print("attack pattern considerated are: " + str(len(attack_ids_dict)))
	return attack_ids_dict, attack_ids_names_dict



def main():
	TACTIC = 'all'

	src = MemoryStore()
	src.load_from_file('../cti/enterprise-attack/enterprise-attack-10.1.json')



	filters = [Filter('type', '=', 'attack-pattern'), Filter("description", "!=", ""), Filter("external_references.source_name", "=", "mitre-attack")]

	attack_patterns = src.query(filters)
	attack_ids_dict, attack_ids_names_dict = get_attack_dict(src, attack_patterns)

	#creation of x and y arrays for the classification task 
	texts_dict = {}

	for ap in attack_patterns:
		ap_parsed = parse(ap, allow_custom= True)
		id = ''
		deprecated_list = get_deprecated_list(src, ap_parsed)
		if ap_parsed.id not in deprecated_list:
			for ref in ap_parsed.external_references:
				if ref.source_name == "mitre-attack":
					id = ref.external_id#map_subtec_to_tec(ref.external_id)
					if id in texts_dict.keys():
						#pass
						old_texts = texts_dict[id]
						old_texts.append(ap_parsed.description)
						texts_dict.update({id: old_texts})
					else:
						texts_dict[id] = [ap_parsed.description]
			for ref in ap_parsed.external_references:
				if ref.source_name == "capec":
					capec_id = ref.external_id
					capec_description = add_description_from_capec(capec_id)
					if id:
						old_texts = texts_dict[id]
						old_texts.extend(capec_description)
						texts_dict.update({id: old_texts})
					#print("[" + ref.external_id + "]->" +ap_parsed.description + "\n")


	relationship_filter = Filter('type', '=', 'relationship') 
	description_filter = Filter("description", "!=", "")
	target_filter = Filter('target_ref', 'contains', 'attack-pattern')
	malware_filter = Filter('source_ref', 'contains', 'malware')
	intrusion_set_filter = Filter('source_ref', 'contains', 'intrusion-set')
	additional_attack_patterns_mal = src.query([relationship_filter, description_filter, target_filter, malware_filter])
	additional_attack_patterns_is = src.query([relationship_filter, description_filter, target_filter, intrusion_set_filter])


	for ap in additional_attack_patterns_mal:					# Added information related to malware objects 
		ap_parsed = parse(ap, allow_custom= True)
		ap_id = ap_parsed.target_ref
		if ap_id in attack_ids_dict.keys():
			attack_id = attack_ids_dict[ap_id]
			if attack_id in texts_dict.keys():
				old_texts = texts_dict[attack_id]
				old_texts.append(ap_parsed.description)
				texts_dict.update({attack_id: old_texts})

	for ap in additional_attack_patterns_is:					# Added information related to instruction-set objects 
		ap_parsed = parse(ap, allow_custom= True)
		ap_id = ap_parsed.target_ref
		if ap_id in attack_ids_dict.keys():
			attack_id = attack_ids_dict[ap_id]
			if attack_id in texts_dict.keys():
				old_texts = texts_dict[attack_id]
				old_texts.append(ap_parsed.description)
				texts_dict.update({attack_id: old_texts})

	print(len(texts_dict))
	#texts_dict is a dictionary data structure and contains for each attack tecnique's id an array of texts that describes it
	#-------------------------------------------------------------------

	#vectors for the DataFrame
	train_texts = []
	train_labels_tec = []
	train_labels_subtec = []
	train_labels_names = []

	for key in texts_dict.keys():
		text = combine_text(texts_dict[key])
		text = cleaning_data(text)
		text = remove_empty_lines(text)
		text = text.strip()
		sentences = sent_tokenize(text)
		for sen in sentences:			
			#sen = re.sub('[%s]' % re.escape(string.punctuation), '', sen)
			if sen not in train_texts:
				train_texts.append(sen)
				train_labels_subtec.append(key)
				new_key = map_subtec_to_tec(key)
				train_labels_tec.append(new_key)
				name = attack_ids_names_dict[key]
				train_labels_names.append(name)

	#Add name of techniques as a new column 

	#DataFrame
	data = {'label_subtec': train_labels_subtec, 'label_tec': train_labels_tec, 'sentence': train_texts, 'tec_name': train_labels_names}
	pd.set_option('max_colwidth',1000)

	data_df = pd.DataFrame(data, columns=['label_tec', 'label_subtec','tec_name', 'sentence'])

	#Saving on file
	data_df.to_csv('dataset_new.csv')#, index=False)


	print(len(train_texts))


	

if __name__ == "__main__":
    main()






