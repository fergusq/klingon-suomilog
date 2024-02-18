import yajwiz

import inflex

from .types import DependencyTreeNode

boqwiz = yajwiz.boqwiz.load_dictionary()

SPECIAL_DICTIONARY = {
	"-nIS:v": "need to",
	"-choH:v": "begin",
	"-qa':v": "again",
	"-moH:v": "make",
	"-laH:v": "can",
	"-taH:v": "still",
	"-neS:v": "my honor",
	"-meH:v": "in order to",
	"-wI':v": "one who",
	"-lu':v": "one",
	"-qu':v": "very",
	"-Ha':v": "un-",

	"-mey:n": "(PLURAL)",
	"-Du':n": "(PLURAL)",
	"-pu':n": "(PLURAL)",
	"-'a':n": "great",
	"-Hom:n": "(DIMINUTIVE)",
	"-oy:n": "dear",
	"-Daq:n": "at",

	"qaStaHvIS": "during",
	"tlhej": "with",

	"'a": "but",
}

SUFFIX_ORDER = {
	"-pa':v": -2,
	"-DI':v": -2,
	"-vIS:v": -2,
	"-chugh:v": -2,
	"-mo':v": -2,
	"-meH:v": -2,

	"-'egh:v": 1,
	"-chuq:v": 1,
	"-qa':v": 1,

	"-Daq:n": -2,
	"-vo':n": -2,
	"-mo':n": -2,
	"-vaD:n": -2,
	"-'e':n": -2,
}

POS_MAP = {
	"NOUN": "N",
	"VERB": "V",
}

def gloss(w: str, pos: str, analyses):
	if w in SPECIAL_DICTIONARY:
		return [SPECIAL_DICTIONARY[w]]

	analyses = [a for a in analyses if not a.get("UNGRAMMATICAL", "") and "{" not in boqwiz.entries.get(a["BOQWIZ_ID"]).definition["en"]]
	if len(analyses) == 0:
		return [w]
	
	if any(a["POS"] == POS_MAP.get(pos, pos) or a["XPOS"] == POS_MAP.get(pos, pos) for a in analyses):
		analyses = [a for a in analyses if a["POS"] == POS_MAP.get(pos, pos) or a["XPOS"] == POS_MAP.get(pos, pos)]

	output = []
	for analysis in analyses:
		morphemes: list[tuple[int, str]] = []
		for part in analysis["PARTS"]:
			position = SUFFIX_ORDER.get(part, 0)
			
			if part in SPECIAL_DICTIONARY:
				morphemes.append((position, SPECIAL_DICTIONARY[part]))
				continue
					
			if part not in boqwiz.entries:
				morphemes.append((position, part[:part.index(":")] if ":" in part else part))
				continue
			
			defn: str = boqwiz.entries[part].definition["en"].lower()
			
			if "-:v" in part:
				subj = defn[:defn.index("-")]
				morphemes.append((-1, subj.strip()))
				obj = defn[defn.rindex("-")+1:]
				if "me" in obj or "us" in obj or "you" in obj:
					morphemes.append((1, obj.strip()))
				continue
			
			if "," in defn:
				defn = defn[:defn.index(",")]
			
			if "(" in defn:
				defn = defn[:defn.index("(")]
											
			if defn.startswith("be "):
				defn = defn[3:]
							
			defn = defn.strip()
			morphemes.append((position, defn))
		
		output.append(" ".join([morpheme for _, morpheme in sorted(morphemes)]))
	
	return output

DEPENDENT_ORDER = {
	# Nouns
	"cc": -3,
	"amod": -2,
	"nmod": -1,
	"acl": 1,
	"conj": 2,

	# Verbs
	"advmod": -3,
	"mark": -2,
	"nsubj": -1,
	"obj": 1,
	"obl": 2,
	"parataxis": 3,
}

def gloss_tree_list(tree: DependencyTreeNode) -> list[list[str]]:
	if tree.pos == "UNK":
		return [[tree.getText()]]

	head_pos = 0
	if tree.pos == "VERB" and tree.token.token in {"qaStaHvIS", "tlhej"}:
		head_pos = -5
	
	head = tree.token.token
	head_glosses = [gloss(head, tree.pos, tree.token.analyses)]
	
	dependents = [(DEPENDENT_ORDER.get(relation, 0), dep.position, gloss_tree_list(dep)) for relation, deps in tree.children.items() for dep in deps]
	dependents.append((head_pos, tree.position, head_glosses))

	dependents.sort()
	dependents = [dep for _, _, deps in dependents for dep in deps]

	return dependents

def get_combinations(left: list[str], right: list[list[str]]):
	if len(right) == 0:
		yield " ".join(left)
		return
	
	for alternative in right[0]:
		yield from get_combinations(left + [alternative], right[1:])

def gloss_tree(tree: DependencyTreeNode):
	dependents = gloss_tree_list(tree)
	return list(get_combinations([], dependents))