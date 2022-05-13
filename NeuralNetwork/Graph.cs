using System.Collections.Immutable;

namespace NeuralNetwork;

public static class Graph
{
	private const double gradient = 6;     // Steepness of sigmoid curve.
	private const double learnRate = 0.05; // Learning rate.

	public delegate ImmutableArray<double> PredictOutputs(ImmutableArray<double> inputs);

	public static PredictOutputs Train(ImmutableArray<int> layerSizes, ImmutableArray<Pattern> patterns, double acceptableError, Random r, int throwIfExceedIterationCount = 2000)
	{
		if (layerSizes.Length < 2)
			throw new ArgumentException("Must be at least two layers - one for input, one for output");
		if (layerSizes.Any(s => s <= 0))
			throw new ArgumentException("All layer sizes must be positive values");
		if (throwIfExceedIterationCount <= 0)
			throw new ArgumentOutOfRangeException(nameof(throwIfExceedIterationCount));
		if (acceptableError < 0)
			throw new ArgumentOutOfRangeException(nameof(acceptableError));
		if (patterns.Any(pattern => (pattern.Inputs.Length != layerSizes.First()) || (pattern.Outputs.Length != layerSizes.Last())))
			throw new ArgumentException("Invalid pattern provided - does not match Input and/or Output sizes");

		// Initialise a network using random weights (in the range -1 to 1)
		var untrainedNetwork = layerSizes.Aggregate(
			seed: ImmutableArray<ImmutableArray<Neuron>>.Empty,
			func: (previousLayers, layerSize) =>
			{
				var previousLayer = previousLayers.LastOrDefault();
				var newLayer = Enumerable.Range(0, layerSize)
					.Select(i => previousLayer.IsDefault
						? Neuron.New(weights: ImmutableArray<double>.Empty)
						: Neuron.New(weights: previousLayer.Select(input => (r.NextDouble() * 2) - 1)))
					.ToImmutableArray();
				return previousLayers.Add(newLayer);
			});

		// Repeatedly run the patterns through the network, propagate the errors back and adjust the weights accordingly until we either reach an acceptable error rate or we
		// exceed the iteration limit
		var trainedStateIfAchievable = Enumerable.Range(0, throwIfExceedIterationCount)
			.Scan(
				seed: new { Error = double.MaxValue, Layers = untrainedNetwork },
				func: (currentState, iteration) => patterns.Aggregate(
					seed: new { Error = 0d, currentState.Layers },
					func: (state, pattern) =>
					{
						// Active the network for the current pattern, then determine the total error for the pattern by comparing the expected pattern outputs to the actual
						// output-layer neuron values. Then propagate the errors back from the outputs to the input layer and adjust the weights based upon these errors.
						var activatedLayers = ActivateNetwork(state.Layers, pattern.Inputs);
						var outputLayer = activatedLayers.Last();
						var errorForActivatedLayers = outputLayer.Zip(pattern.Outputs, (neuron, patternOutput) => Math.Pow(neuron.GetOutput() - patternOutput, 2)).Sum();
						return new
						{
							Error = state.Error + errorForActivatedLayers,
							Layers = AdjustWeightsThroughNetwork(PropagateErrorBackThroughNetwork(activatedLayers, pattern.Outputs))
						};
					}))
			.Select((state, iteration) =>
			{
				// The first "iteration" will be the initialisation state, so state.Error will double.MaxValue - there is no point displaying that
				if (iteration > 0)
					Console.WriteLine(DateTime.Now.ToString("HH:mm:ss.fff") + " Iteration {0}\tError {1:0.000}", iteration, state.Error);
				return state;
			})
			.FirstOrDefault(state => state.Error <= acceptableError);
		if (trainedStateIfAchievable is null)
			throw new ReachedLocalErrorMinimumException();

		return inputs =>
			inputs.Length != trainedStateIfAchievable.Layers.First().Length
				? throw new ArgumentException("Invalid number of inputs")
				: ActivateNetwork(trainedStateIfAchievable.Layers, inputs)
					.Last()
					.Select(neuron => neuron.GetOutput());
	}

	// "Activate" the network by setting the input values to those from a specified pattern and then using the current weight values to calculate the second layer's
	// neuron values and then the third layer until the next iteration is fully constructed. A complete iteration is an array of arrays of Neurons as there are multiple
	// layers and each layer consists of a list of neurons.
	//
	// The Scan call iterates over each layer and looks at the previous layer (if there is one) to determine what values each neuron in the current layer should have:
	//  1. if there is no previous layer then it's the input layer and the 'patternInputs' are used to set the neuron values
	//  2. if there IS a previous layer then the neuron	values are calculated by looking at all of the weighted connections between each neuron and the ones connected
	//     to it in the previous layer.
	private static ImmutableArray<ImmutableArray<Neuron>> ActivateNetwork(ImmutableArray<ImmutableArray<Neuron>> layers, ImmutableArray<double> patternInputs) =>
		layers
			.Scan(
				seed: default(ImmutableArray<Neuron>),
				func: (previousLayer, layer) => previousLayer.IsDefault
					? layer.Zip(patternInputs, (neuron, patternInput) => neuron with { RawOutput = patternInput })
					: layer.Select(neuron => Activate(neuron, previousLayer)))
			.Skip(1) // Skip the null value that will appear at the start of the list (the seed value from Scan call)
			.ToImmutableArray();

	private static Neuron Activate(Neuron neuron, ImmutableArray<Neuron> previousLayer) =>
		previousLayer.Length != neuron.WeightsOfConnectionsToPreviousLayer.Length
			? throw new Exception($"The number of neurons in {nameof(previousLayer)} must match the number of entries in the neuron's {nameof(neuron.WeightsOfConnectionsToPreviousLayer)}")
			: neuron with
			{
				Error = 0,
				Input = neuron.WeightsOfConnectionsToPreviousLayer
					.Select((weight, index) => weight * previousLayer[index].GetOutput())
					.Sum()
			};

	// Given a list of expected outputs for the network, set error values on the neurons in each layer based upon how far off their current values are.
	//
	// The ScanBack call works backwards from the output layer through to the input layer:
	//  1. if there is no 'layerToPropagateErrorBackFrom' (ie. looking at the output layer) then calculate error by comparing the neuron's value to the expected output
	//  2. if there IS a 'layerToPropagateErrorBackFrom' then calculate an error by taking all of the connections from the current neuron to neurons in the layer ahead
	//     of it, multiplying each of those neurons' error values by the weight of the connection and them summing those values
	private static ImmutableArray<ImmutableArray<Neuron>> PropagateErrorBackThroughNetwork(ImmutableArray<ImmutableArray<Neuron>> layers, ImmutableArray<double> patternOutputs) =>
		layers
			.ScanBack(
				seed: default(ImmutableArray<Neuron>),
				func: (layer, layerToPropagateErrorBackFrom) => layerToPropagateErrorBackFrom.IsDefault
					? layer.Zip(patternOutputs, (neuron, patternOutput) => AddToCurrentError(neuron, patternOutput - neuron.GetOutput()))
					: layerToPropagateErrorBackFrom.Aggregate(
						seed: layer,
						func: (layerToPropagateErrorTo, neuron) => PropagateErrorToPreviousLayer(neuron, layerToPropagateErrorTo)))
			.Take(layers.Length) // Skip the null value that will appear at the end of the list (the seed value from ScanBack call)
			.ToImmutableArray();

	private static ImmutableArray<Neuron> PropagateErrorToPreviousLayer(Neuron neuron, ImmutableArray<Neuron> previousLayer) =>
		(previousLayer.Length != neuron.WeightsOfConnectionsToPreviousLayer.Length)
			? throw new Exception($"The number of neurons in {nameof(previousLayer)} must match the number of entries in the neuron's {nameof(neuron.WeightsOfConnectionsToPreviousLayer)}")
			: neuron.WeightsOfConnectionsToPreviousLayer.Select((weight, index) =>
			{
				var otherNeuron = previousLayer[index];
				var delta = neuron.Error * weight;
				return AddToCurrentError(otherNeuron, delta);
			});

	private static Neuron AddToCurrentError(Neuron neuron, double delta) => neuron with { Error = neuron.Error + delta };

	// Adjust the connection weights between layers through the network, based upon the current error values that each neuron has.
	//
	// Work forwards through the layers:
	//  1. The first layer doesn't need any changes as it doesn't have any weighted connections to neurons in a previous layer 
	//  2. Subsequent layers update the weights of their connections by considering the output values of the neurons in the previous layer, the Error value of the
	//     current neuron and the learnRate constant
	private static ImmutableArray<ImmutableArray<Neuron>> AdjustWeightsThroughNetwork(ImmutableArray<ImmutableArray<Neuron>> layers)
	{
		// Construct a 'previousLayers' array, with an entry corresponding to each layer of the network - the first value will be null as there is no previous layer
		// for the first (aka. input) layer of the network
		var previousLayers = ImmutableArray<ImmutableArray<Neuron>>.Empty
			.Add(default)
			.AddRange(layers.Take(layers.Length - 1));

		return layers.Zip(
			previousLayers,
			(layer, previousLayer) => previousLayer.IsDefault
				? layer
				: layer.Select(neuron => AdjustWeightsForCurrentError(neuron, previousLayer)));
	}

	private static Neuron AdjustWeightsForCurrentError(Neuron neuron, ImmutableArray<Neuron> previousLayer)
	{
		if (previousLayer.Length != neuron.WeightsOfConnectionsToPreviousLayer.Length)
			throw new Exception($"The number of neurons in {nameof(previousLayer)} must match the number of entries in the neuron's {nameof(neuron.WeightsOfConnectionsToPreviousLayer)}");

		var activation = neuron.GetOutput();
		var derivative = activation * (1 - activation);
		return neuron with
		{
			Bias = neuron.Bias + (neuron.Error * derivative * learnRate),
			WeightsOfConnectionsToPreviousLayer = neuron.WeightsOfConnectionsToPreviousLayer
				.Select((weight, index) => weight + neuron.Error * derivative * learnRate * previousLayer[index].GetOutput())
		};
	}

	/// <summary>
	/// This will be thrown if the training can not reduce its total error to the specified acceptableError value - presumably the training process has got caught
	/// in a local maximum that it can not escape from, trying again with a different Random instance may produce better results by starting with different weights
	/// in the untrained network
	/// </summary>
	public sealed class ReachedLocalErrorMinimumException : Exception { }

	private sealed record Neuron(double Bias, double Error, double Input, double RawOutput, ImmutableArray<double> WeightsOfConnectionsToPreviousLayer)
    {
		public static Neuron New(ImmutableArray<double> weights) => new(Bias: 0, Error: 0, Input: 0, RawOutput: double.MinValue, weights);

		public double GetOutput() =>
			RawOutput != double.MinValue
				? RawOutput
				: 1 / (1 + Math.Exp(-gradient * (Input + Bias)));
	}
}