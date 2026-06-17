from SpaceNet.Plugins.multiplexer import Multiplexer
from SpaceNet.Plugins.plugin import Link
from typing import Any, TypeAlias
from SpaceNet.Plugins.plugin import Ports
from SpaceNet.Plugins.plugin import Plugin
from SpaceNet.Utils.id import ID
from dataclasses import dataclass


@dataclass
class Connection:
    source_node: ID
    source_port: ID
    target_node: ID
    target_port: ID


Nodes: TypeAlias = dict[ID, Plugin]
Connections: TypeAlias = list[Connection]


class Graph:


    def __init__(self):
        self.nodes: Nodes = {"input": Multiplexer(),
                             "output": Multiplexer()}
        self.connections: Connections = []
        self.is_compiled: bool = False


    def register_node(self, identifier: ID, plugin: Plugin):

        self.nodes[identifier] = plugin


    def connect_node(self, source_node: ID, source_port: ID, target_node: ID, target_port: ID):

        self.connections.append(Connection(source_node, source_port, target_node, target_port))


    def evaluate(self, **inputs) -> dict[ID, Any]:

        # provide a fresh Link / input data for the connected nodes
        for id, value in inputs.items():
            self.nodes["input"].inputs[id] = Link(value)

        self.compile()
        self._execute_graph()

        return { id: output.value for id, output in self.nodes["output"].outputs.items() }


    def compile(self):

        if not self.is_compiled:
            self._build_graph()
            self._sort_graph()
            self.is_compiled = True


    def _build_graph(self):

        for connection in self.connections:
            source_node, source_port, target_node, target_port = connection.source_node, connection.source_port, connection.target_node, connection.target_port

            link = self.nodes[source_node].outputs.setdefault(source_port, Link())
            self.nodes[target_node].inputs[target_port] = link

            # set the input recipe
            if source_node == "input":
                self.nodes["input"].inputs.setdefault(source_port, Link())
            # set the output recipe
            if target_node == "output":
                self.nodes["output"].outputs.setdefault(target_port, Link())


    def _sort_graph(self):

        fulfilled_nodes = {"input"}
        sorted_nodes = { "input": self.nodes["input"] }
        processing_nodes = [
            node_id
            for node_id in self.nodes
            if node_id not in {"input", "output"}
        ]

        while len(sorted_nodes) < len(processing_nodes) + 1: # because the special input node is the first node in the sorted list
            next_node = next(
                (
                    node_id
                    for node_id in processing_nodes
                    if node_id not in sorted_nodes
                    # does the node act as target for any other node?
                    and any(
                        connection.target_node == node_id
                        for connection in self.connections
                    )
                    # when the node act as a target node, check if all its source nodes has been fulfilled
                    and all(
                        connection.source_node in fulfilled_nodes
                        for connection in self.connections
                        if connection.target_node == node_id
                    )
                ),
                None,
            )

            if next_node is None:
                raise ValueError(
                    "Recipe graph cannot be sorted: nodes are cyclic, disconnected, or depend on an unknown source"
                )

            sorted_nodes[next_node] = self.nodes[next_node]
            fulfilled_nodes.add(next_node)

        sorted_nodes["output"] = self.nodes["output"]
        self.nodes = sorted_nodes


    def _execute_graph(self):

        for node in self.nodes.values():
            node.execute()
