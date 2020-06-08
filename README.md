# dash-inventory-flow

A dashboard for monitoring inventory imbalances, written in Python. It uses the [Dash](https://plotly.com/dash/) framework for building visuals, [OSmnx](https://osmnx.readthedocs.io/en/stable/) and [NetworkX](https://networkx.github.io/) for the mathematical solution.

## How it works

It generates random supply and demand data for an item, in random locations. The process under the dashboard solves the [Multi-commodity flow problem](https://en.wikipedia.org/wiki/Multi-commodity_flow_problem) and suggest a set of routes (with minimal total cost) that minimize supply or demand imbalances. The problem is solved for a single item/commodity but can easily generalized.

## Deployment

Use the `Dockerfile` to create and run the environment:

```
$CONTAINER_NAME = <your-container-name>
$ docker build -t $CONTAINER_NAME .
$ docker run -d $CONTAINER_NAME
```