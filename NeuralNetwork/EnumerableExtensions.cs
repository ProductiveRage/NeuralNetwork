namespace NeuralNetwork;

public static class EnumerableExtensions
{
	/// <summary>
	/// This will enumerate a sequence and thread a value through it - the first value that will be returned will be the initial seed value, the second value will be
	/// the result of the func operating on the seed and the first value of the input sequence, the third value will be the result of the func operating on the last
	/// generated value and the second value of the input sequence.
	/// 
	/// This is similar in principle to Aggregate except that Aggregate processes the entire sequence in one go and returns a single value whereas this one returns
	/// each interim value in a sequence, meaning that the sequence may be terminated early if desired.
	/// </summary>
	public static IEnumerable<TAccumulate> Scan<TSource, TAccumulate>(this IEnumerable<TSource> source, TAccumulate seed, Func<TAccumulate, TSource, TAccumulate> func)
	{
		yield return seed; // F# will return the seed value before considering any further values

		var nextValue = seed;
		foreach (var value in source)
		{
			nextValue = func(nextValue, value);
			yield return nextValue;
		}
	}

	/// <summary>
	/// This is similar to Scan except that it works from the end of the input sequence and traverses backwards. While this returns a sequence of interim results,
	/// like Scan, it can not be terminated part way through the enumeration as all the entire sequence must be evaluated before the first value can be returned.
	/// </summary>
	public static IEnumerable<TAccumulate> ScanBack<TSource, TAccumulate>(this IEnumerable<TSource> source, TAccumulate seed, Func<TSource, TAccumulate, TAccumulate> func)
	{
		var sourceArray = source.ToArray();
		var results = new Stack<TAccumulate>(capacity: sourceArray.Length + 1);
		results.Push(seed);
		
		var nextValue = seed;
		foreach (var previousValue in source.Reverse())
		{
			nextValue = func(previousValue, nextValue);
			results.Push(nextValue);
		}
		return results;
	}
}