using System.Collections.Immutable;
using NeuralNetwork;

var filename = "iris_data.csv";
var numberOfInputs = 4;
var numbersOfNeuronsInHiddenLayers = new[] { 4, 3, 2 };
var numberOfOutputs = 3;
var acceptableError = 2;

var patterns = (await File.ReadAllLinesAsync(filename))
    .Where(line => !string.IsNullOrWhiteSpace(line) && !line.Trim().StartsWith('#'))
    .Select(line =>
    {
        var values = line.Split(',').Select(double.Parse);
        return new Pattern(
            inputs: values.Take(numberOfInputs).ToImmutableArray(),
            outputs: values.Skip(numberOfInputs).ToImmutableArray());
    })
    .ToImmutableArray();

var result = Graph.Train(
    layerSizes: numbersOfNeuronsInHiddenLayers.Prepend(numberOfInputs).Append(numberOfOutputs).ToImmutableArray(),
    patterns,
    acceptableError,
    r: new Random(0));

Console.WriteLine("Done! Press [Enter] to terminate..");
Console.ReadLine();