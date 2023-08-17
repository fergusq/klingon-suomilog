from collections import defaultdict
import pprint
import readline
from pathlib import Path
import traceback
from typing import DefaultDict, Dict, List, NamedTuple, Set, Tuple, Union

import suomilog.patternparser as pp
from suomilog.cykparser import CYKParser
import yajwiz

def match_bits(tbits: Set[str], bits: Set[str]):
	positive_bits = {bit for bit in bits if not bit.startswith("!")}
	negative_bits = {bit[1:] for bit in bits if bit.startswith("!")}
	return tbits >= positive_bits and not (tbits & negative_bits)

class MyToken(pp.Token):
	def __init__(self, token: str, alternatives: List[Tuple[str, Set[str]]], position: int, analyses: List[yajwiz.analyzer.Analysis]):
		super().__init__(token, alternatives)
		self.position = position
		self.analyses = analyses
	def containsMatch(self, alternatives):
		return any([any([tbf == baseform and match_bits(tbits, bits) for tbf, tbits in self.alternatives]) for baseform, bits in alternatives])

def tokenize(line) -> List[MyToken]:
	ans = []
	i = 0
	for token_type, token_text in yajwiz.tokenize(line):
		if token_type == "WORD":
			analyses = yajwiz.analyze(token_text, include_syntactical_info=True)
			alternatives = []
			for analysis in analyses:
				bits = analysis.get("SYNTAX_INFO", {}).get("BITS", set())
				alternatives.append((analysis["LEMMA"], bits))
			
			if not analyses:
				if len(token_text) > 3 and token_text.endswith("vaD"):
					alternatives.append((token_text[:-3], {f"«{token_text}»", token_text[:-3], "N", "NP", "-vaD", "-vaD:n", "N5"}))
				
				elif len(token_text) > 3 and token_text.endswith("mo'"):
					alternatives.append((token_text[:-3], {f"«{token_text}»", token_text[:-3], "N", "NP", "-mo'", "-mo':n", "N5"}))
				
				elif len(token_text) > 3 and token_text.endswith("vo'"):
					alternatives.append((token_text[:-3], {f"«{token_text}»", token_text[:-3], "N", "NP", "-vo'", "-vo':n", "N5"}))
				
				elif len(token_text) > 3 and token_text.endswith("Daq"):
					alternatives.append((token_text[:-3], {f"«{token_text}»", token_text[:-3], "N", "NP", "-Daq", "-Daq:n", "N5"}))
				
				elif len(token_text) > 3 and token_text.endswith("'e'"):
					alternatives.append((token_text[:-3], {f"«{token_text}»", token_text[:-3], "N", "NP", "-'e'", "-'e':n", "N5"}))
				
				else:
					alternatives.append((token_text, {f"«{token_text}»", token_text, "N", "NP"}))
			
			ans.append(MyToken(token_text, alternatives, i, analyses))
			i += 1
		
		elif token_type == "PUNCT":
			if token_text in {",", ":", ";"}:
				ans.append(MyToken(",", [], i, []))
				i += 1

	
	return ans

grammar = pp.Grammar()

class WordPattern:
	def __init__(self, bits: Set[str] = set()):
		self.bits = set() | bits
	def __repr__(self):
		return f"WordPattern({repr(self.bits)})"
	def toCode(self):
		if not self.bits:
			return "<pattern that matches any single token>"
		
		else:
			return f"<pattern that matches any single token with bits {{{','.join(self.bits)}}}>"
	def match(self, grammar, tokens, bits):
		bits = self.bits|bits
		if len(tokens) != 1 or bits and not any(match_bits(altbits, bits) for _, altbits in tokens[0].alternatives):
			return []
		return [tokens[0].token]
	def allowsEmptyContent(self):
		return False
	def expandBits(self, name, grammar, expanded, bits):
		return WordPattern(self.bits|bits)

grammar.patterns["."] = [
	WordPattern()
]

class MyStringOutput(pp.Output):
	def __init__(self, string: str):
		self.string = string
	def __repr__(self):
		return "StringOutput(" + repr(self.string) + ")"
	def eval(self, args):
		args = [f"node({repr(arg.position)}, {repr(arg.token)})" if isinstance(arg, MyToken) else str(arg) for arg in args]
		ans = self.string
		for i, a in enumerate(args):
			var = "$"+str(i+1)
			if var not in ans:
				ans = ans.replace("$*", ",".join(args[i:]))
				break
			ans = ans.replace(var, str(a))
		return ans

query = pp.Pattern("QUERY", [
	pp.PatternRef("SENTENCE", set())
], pp.StringOutput("sentence($1)"))

path = Path("grammar.suomilog")
with path.open("r") as file:
	for line in file:
		if "::=" in line:
			grammar.parseGrammarLine(line.replace("\n", ""), default_output=MyStringOutput)

n_patterns = sum([len(category) for category in grammar.patterns.values()])
print("Ladattu", n_patterns, "fraasia.")



cyk_parser = CYKParser(grammar, "SENTENCE")
#cyk_parser.print()

pp.setDebug(0)
tokens = []
analysis = None
brackets = False

identity = lambda x: x
def set_pos(pos):
	return lambda node: node._replace(pos=pos)

tree = Union["node", "branch"]

class node(NamedTuple):
	position: int
	text: str
	pos: str = "UNK"

	def getPosition(self) -> int:
		return self.position
	
	def getRelations(self) -> Dict[int, Tuple[int, str]]:
		return {}
	
	def getInverseRelations(self) -> Dict[int, Dict[str, Set[int]]]:
		return {}
	
	def getPartsOfSpeech(self):
		return {self.position: self.pos}
	
	def getWeight(self):
		return 0

class branch:
	attrs: Dict[str, tree]
	weight: float
	def __init__(self, weight=0, **x):
		self.attrs = x
		self.weight = weight
	
	def getPosition(self) -> int:
		return self.attrs["head"].getPosition()
	
	def getRelations(self) -> Dict[int, Tuple[int, str]]:
		relations = self.attrs["head"].getRelations()
		position = self.getPosition()
		for key, val in self.attrs.items():
			if key != "head":
				relations = {**relations, **val.getRelations()}
				relations[val.getPosition()] = (position, key)
		
		return relations
	
	def getInverseRelations(self) -> Dict[int, Dict[str, Set[int]]]:
		ans = defaultdict(lambda: defaultdict(set))
		for a, (b, rel) in self.getRelations().items():
			ans[b][rel].add(a)
		
		return {key: dict(val) for key, val in ans.items()}
	
	def getPartsOfSpeech(self):
		poses = {}
		for val in self.attrs.values():
			poses.update(val.getPartsOfSpeech())
		
		return poses
	
	def getWeight(self):
		w = self.weight
		for val in self.attrs.values():
			w += val.getWeight()
		
		return w

def validateInverseRelations(invrels: Dict[int, Dict[str, Set[int]]]) -> bool:
	for position, relations in invrels.items():
		if "ccomp" in relations and "advmod" in relations and any(n < list(relations["ccomp"])[0] for n in relations["advmod"]):
			return False
		if "ccomp" in relations and "obl" in relations and any(n < list(relations["ccomp"])[0] for n in relations["obl"]):
			return False
	
	return True

CONSTITUENT_CLASSES = {
	"node": node,

	"NOUN": set_pos("NOUN"),
	"VERB": set_pos("VERB"),
	"CONJ": set_pos("CONJ"),
	"ADV": set_pos("ADV"),
	"QUES": set_pos("QUES"),

	"CAUSE": identity,
	"COPULA": identity,
	"FOR": identity,
	"FROM": identity,
	"TOPIC": identity,
	"PLACE": identity,
	"TIME": identity,

	"ALSOP": branch,
	"ONLYP": branch,
	"COP": branch,
	"RELP": branch,
	"SS": branch,
	"NP": branch,
	"S": branch,
	"VP": branch,
}

while True:
	try:
		line = input(">> ").strip()
	except EOFError:
		print()
		break
	if not line:
		continue
	elif line[0] == ".":
		line = line.replace("\t", " ")
		tokens = line.split(" ")
		if len(tokens) == 1 and tokens[0][1:] in grammar.patterns:
			print(grammar.patterns[tokens[0][1:]])
		elif len(tokens) > 2 and tokens[1] == "::=" and "->" in tokens:
			grammar.parseGrammarLine(line)
		else:
			print("Kategoriat:")
			for cat in grammar.patterns:
				print(cat)
	elif line.startswith("/eval "):
		try:
			print(eval(line[len("/eval "):]))
		except:
			traceback.print_exc()
	elif line == "/cyk_table":
		pprint.pprint(analysis.cyk_table)
		print("<table>")
		for span in range(len(tokens), 0, -1):
			print("<tr>", end="")
			for start in range(0, len(tokens)-span+1):
				end = start+span
				print(f"<td>{repr(analysis.cyk_table[(start, end)])}", end="")
			
			print()
		
		print("<tr>", end="")
		for t in tokens:
			print(f"<th>{t.token}", end="")
		
		print()
		print("</table>")
	elif line == "/säännöt":
		cyk_parser.print()
	elif line == "/hakasulut":
		brackets = not brackets
	else:
		tokens = tokenize(line)
		print(tokens)
		analysis = cyk_parser.parse(tokens)
		#pprint.pprint(cyk_table)
		#pprint.pprint(split_table)
		outputs = analysis.getOutput(".SENTENCE{}")
		if not outputs:
			continue
		for output in outputs:
			print(output)
			dep_tree: tree = eval(output.strip(), CONSTITUENT_CLASSES)
			if not validateInverseRelations(dep_tree.getInverseRelations()):
				continue
			weight = dep_tree.getWeight()
			relations = dep_tree.getRelations()
			poses = dep_tree.getPartsOfSpeech()
			root_position = dep_tree.getPosition()
			print("WEIGHT:", weight)
			for token in tokens:
				head, rel = relations.get(token.position, (-1, "unk") if token.position != root_position else (-1, "root"))
				print(token.position, token.token, poses.get(token.position, "UNK"), head, rel, sep="\t")
			print("digraph G {")
			for token in tokens:
				print(f"   n{token.position} [label=\"{token.token}\"];")
			print(f"   n{dep_tree.getPosition()} -> ROOT [label=\"root\"];")
			for a, (b, rel) in relations.items():
				print(f"   n{a} -> n{b} [label=\"{rel}\"];")
			print("}")
