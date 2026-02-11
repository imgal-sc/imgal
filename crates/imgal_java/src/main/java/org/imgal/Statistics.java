package org.imgal;

import java.lang.foreign.Arena;
import java.lang.foreign.FunctionDescriptor;
import java.lang.foreign.Linker;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;
import java.lang.invoke.MethodHandle;

/**
 * Interface for native statistics.
 *
 * @author Edward Evans
 */
public class Statistics extends AbstractNativeLibrary {
	// declare method handles
	static final MethodHandle sumMH = initSumMH();

	/**
	 * Compute the sum of an array.
	 *
	 * @param input The input array to sum.
	 * @return The sum of the array.
	 */
	public static double sum(double[] input) throws Throwable {
		try (Arena arena = Arena.ofConfined()) {
			// allocate memory for the array and obtain array length
			MemorySegment arr = MemorySegment.ofArray(input);
			long len = (long) input.length;

			return (double) sumMH.invokeExact(arr, len); }
	}

	/**
	 * Initialize a MethodHandle for imgal::statistics::sum.
	 */
	private static MethodHandle initSumMH() {
		// locate the function symbol
		MemorySegment fnMS = libLookup.find("sum").orElseThrow();

		// create a function descriptor
		FunctionDescriptor fnSig = FunctionDescriptor.of(
				ValueLayout.JAVA_DOUBLE,
				ValueLayout.ADDRESS,
				ValueLayout.JAVA_LONG);

		// return the downcall handle
		return linker.downcallHandle(fnMS, fnSig, Linker.Option.critical(true));
	}
}
