import yajwiz

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

EARLIED = {
	"-'egh:v",
	"-chuq:v",
	"-qa':v",
}

DEFERRED = {
	"-pa':v",
	"-DI':v",
	"-vIS:v",
	"-chugh:v",
}

POS_MAP = {
	"NOUN": "N",
	"VERB": "V",
}

def gloss(w: str, pos: str, analyses):
	if w in SPECIAL_DICTIONARY:
		return SPECIAL_DICTIONARY[w]

	analyses = [a for a in analyses if not a.get("UNGRAMMATICAL", "") and "{" not in boqwiz.entries.get(a["BOQWIZ_ID"]).definition["en"]]
	if len(analyses) == 0:
		return w
	
	if any(a["POS"] == POS_MAP.get(pos, pos) or a["XPOS"] == POS_MAP.get(pos, pos) for a in analyses):
		analyses = [a for a in analyses if a["POS"] == POS_MAP.get(pos, pos) or a["XPOS"] == POS_MAP.get(pos, pos)]

	output = []
	for analysis in analyses:
		deferred = []
		middle = []
		earlied = []
		for part in analysis["PARTS"]:
			place = middle
			if part in EARLIED:
				place = earlied
			if part in DEFERRED:
				place = deferred
			
			if part in SPECIAL_DICTIONARY:
				place.append(SPECIAL_DICTIONARY[part])
				continue
					
			if part not in boqwiz.entries:
				place.append(part[:part.index(":")] if ":" in part else part)
				continue
			
			defn = boqwiz.entries[part].definition["en"].lower()
			
			if "-:v" in part:
				defn = defn[:defn.index("-")]
				deferred.append(defn.strip())
				continue
			
			if "," in defn:
				defn = defn[:defn.index(",")]
			
			if "(" in defn:
				defn = defn[:defn.index("(")]
											
			if defn.startswith("be "):
				defn = defn[3:]
							
			defn = defn.strip()
			place.append(defn)
						
		output.append(" ".join(reversed(earlied + middle + deferred)))
	
	return " / ".join(output)