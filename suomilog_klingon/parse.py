from collections import defaultdict
import pprint
import readline
from pathlib import Path
import traceback
from typing import NamedTuple, Union

import suomilog.patternparser as pp
from suomilog.cykparser import CYKParser
import yajwiz

from . import glosser as glosser
from .types import MyToken, match_bits, tree, node, branch, set_pos, identity, validateInverseRelations, DependencyTreeNode


def tokenize(line) -> list[MyToken]:
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
				# Simple guesser
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
	def __init__(self, bits: set[str] = set()):
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

CONSTITUENT_CLASSES = {
	"node": node,

	"NOUN": set_pos("NOUN"),
	"VERB": set_pos("VERB"),
	"CONJ": set_pos("CONJ"),
	"ADV": set_pos("ADV"),
	"QUES": set_pos("QUES"),
	"EXCL": set_pos("EXCL"),

	"CAUSE": identity,
	"COPULA": identity,
	"FOR": identity,
	"FROM": identity,
	"TOPIC": identity,
	"PLACE": identity,
	"TIME": identity,

	"ALSOP": branch.with_name("ALSOP"),
	"ONLYP": branch.with_name("ONLYP"),
	"COP": branch.with_name("COP"),
	"RELP": branch.with_name("RELP"),
	"SS": branch.with_name("SS"),
	"NP": branch.with_name("NP"),
	"S": branch.with_name("S"),
	"VP": branch.with_name("VP"),
}

def printConstituencyTree(t: tree):
	queue = []  # (node, parent, is_last)
	queue.append((t, "ROOT", True))
	while queue:
		t, parent, is_last = queue.pop(0)
		if isinstance(t, str):
			print(" " + t + " ", end="")
			queue.append((" "*len(t), parent, is_last))

		else:
			width = t.getTextLength()
			if isinstance(t, node):
				print(" " + t.text + " ", end="")
				queue.append((t.pos + " "*(width-len(t.pos)-2), t.text, is_last))
			
			else:
				children = list(t.attrs.keys())
				if len(children) == 1 and isinstance(children[0], branch) and children[0].name == parent:
					queue.append((t.attrs[children[0]], parent, is_last))
					continue

				print(" " + t.name + "-"*(width - len(t.name) - 2) + " ", end="")
				for i, child in enumerate(children):
					queue.append((t.attrs[child], child, is_last and i == len(children)-1))
		
		if is_last:
			print()
		
		if all(isinstance(x, str) and x.strip() == "" for x, _, _ in queue):
			print()
			break


def parse(tokens: list[MyToken]) -> list[tree]:
	analysis = cyk_parser.parse(tokens)
	outputs = analysis.getOutput(".SENTENCE{}")
	if not outputs:
		return []

	dep_trees = []
	for output in outputs:
		dep_tree: tree = eval(output.strip(), CONSTITUENT_CLASSES)
		if not validateInverseRelations(dep_tree):
			continue
		dep_trees.append(dep_tree)
	
	dep_trees.sort(key=lambda x: x.getWeight())
	return dep_trees

if __name__ == "__main__":
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
		# elif line == "/cyk_table":
		# 	pprint.pprint(analysis.cyk_table)
		# 	print("<table>")
		# 	for span in range(len(tokens), 0, -1):
		# 		print("<tr>", end="")
		# 		for start in range(0, len(tokens)-span+1):
		# 			end = start+span
		# 			print(f"<td>{repr(analysis.cyk_table[(start, end)])}", end="")
				
		# 		print()
			
		# 	print("<tr>", end="")
		# 	for t in tokens:
		# 		print(f"<th>{t.token}", end="")
			
		# 	print()s
		# 	print("</table>")
		elif line == "/säännöt":
			cyk_parser.print()
		else:
			tokens = tokenize(line)
			print(tokens)
			dep_trees = parse(tokens)
			for dep_tree in dep_trees:
				print(repr(dep_tree))
				printConstituencyTree(dep_tree)
				weight = dep_tree.getWeight()
				relations = dep_tree.getRelations()
				poses = dep_tree.getPartsOfSpeech()
				root_position = dep_tree.getPosition()
				print("WEIGHT:", weight)
				for token in tokens:
					head, rel = relations.get(token.position, (-1, "unk") if token.position != root_position else (-1, "root"))
					pos = poses.get(token.position, "UNK")
					print(token.position, token.token, pos, head, rel, " / ".join(glosser.gloss(token.token, pos, token.analyses)), sep="\t")
				print("digraph G {")
				for token in tokens:
					print(f"   n{token.position} [label=\"{token.token}\"];")
				print(f"   n{dep_tree.getPosition()} -> ROOT [label=\"root\"];")
				for a, (b, rel) in relations.items():
					print(f"   n{a} -> n{b} [label=\"{rel}\"];")
				print("}")
				print(glosser.gloss_tree(DependencyTreeNode.fromTree(tokens, dep_tree)))
