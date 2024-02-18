from collections import defaultdict
from typing import NamedTuple, Union

import suomilog.patternparser as pp
import yajwiz.analyzer

def match_bits(tbits: set[str], bits: set[str]):
	positive_bits = {bit for bit in bits if not bit.startswith("!")}
	negative_bits = {bit[1:] for bit in bits if bit.startswith("!")}
	return tbits >= positive_bits and not (tbits & negative_bits)

class MyToken(pp.Token):
	def __init__(self, token: str, alternatives: list[tuple[str, set[str]]], position: int, analyses: list[yajwiz.analyzer.Analysis]):
		super().__init__(token, alternatives)
		self.position = position
		self.analyses = analyses
	def containsMatch(self, alternatives):
		return any([any([tbf == baseform and match_bits(tbits, bits) for tbf, tbits in self.alternatives]) for baseform, bits in alternatives])


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
	
	def getRelations(self) -> dict[int, tuple[int, str]]:
		return {}
	
	def getInverseRelations(self) -> dict[int, dict[str, set[int]]]:
		return {}
	
	def getPartsOfSpeech(self):
		return {self.position: self.pos}
	
	def getWeight(self):
		return 0
	
	def getTextLength(self):
		return len(self.text) + 2

class branch:
	name: str
	attrs: dict[str, tree]
	weight: float
	def __init__(self, name="", weight=0, **x):
		self.name = name
		self.attrs = x
		self.weight = weight
	
	def getPosition(self) -> int:
		return self.attrs["head"].getPosition()
	
	def getRelations(self) -> dict[int, tuple[int, str]]:
		relations = self.attrs["head"].getRelations()
		position = self.getPosition()
		for key, val in self.attrs.items():
			if key != "head":
				relations = {**relations, **val.getRelations()}
				relations[val.getPosition()] = (position, key)
		
		return relations
	
	def getInverseRelations(self) -> dict[int, dict[str, set[int]]]:
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
	
	def getTextLength(self):
		return sum(val.getTextLength() for val in self.attrs.values())
	
	def __repr__(self):
		return f"branch({self.name}, {self.weight}, **{self.attrs})"
	
	@staticmethod
	def with_name(name):
		return lambda **x: branch(name=name, **x)

def validateInverseRelations(t: tree) -> bool:
	invrels: dict[int, dict[str, set[int]]]
	invrels = t.getInverseRelations()
	for position, relations in invrels.items():
		# Check that there are no adverbial elements before a clausal complement
		# I.e. DaH jIDoy' 'e' vItlhoj cannot be interpreted as "Now I realized that I am tired", only as "I realize that I am tired now"
		if "ccomp" in relations and "advmod" in relations and any(n < list(relations["ccomp"])[0] for n in relations["advmod"]):
			return False
		if "ccomp" in relations and "obl" in relations and any(n < list(relations["ccomp"])[0] for n in relations["obl"]):
			return False
	
	return True


class DependencyTreeNode(NamedTuple):
	position: int
	pos: str
	token: MyToken
	children: dict[str, list["DependencyTreeNode"]]

	@staticmethod
	def fromTree(tokens: list[MyToken], tree: tree) -> "DependencyTreeNode":
		relations = tree.getRelations()
		poses = tree.getPartsOfSpeech()
		root_position = tree.getPosition()

		nodes: dict[int, DependencyTreeNode] = {}
		for token in tokens:
			nodes[token.position] = DependencyTreeNode(token.position, poses.get(token.position, "UNK"), token, {})
		
		for a, (b, rel) in relations.items():
			nodes[b].children.setdefault(rel, []).append(nodes[a])
		
		return nodes[root_position]
	
	def getText(self) -> str:
		tokens = sorted(self.getTokens(), key=lambda t: t.position)
		return " ".join(t.token for t in tokens)

	def getTokens(self):
		yield self.token
		for child in self.children.values():
			for node in child:
				yield from node.getTokens()