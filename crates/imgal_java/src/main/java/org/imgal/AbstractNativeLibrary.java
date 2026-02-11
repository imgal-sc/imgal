package org.imgal;

import java.lang.foreign.Arena;
import java.lang.foreign.Linker;
import java.lang.foreign.SymbolLookup;
import java.io.InputStream;
import java.net.URL;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.StandardCopyOption;

/**
 * Abstract class for all Java bindings.
 *
 * @author Edward Evans
 */
public abstract class AbstractNativeLibrary {
	private static final String libName = "imgal_c";
	private static final String libPath;
	private static final String libPrefix;
	private static final String libExtension;
	private static final String fileName;
	public static final SymbolLookup libLookup;
	public static final Linker linker = Linker.nativeLinker();


	// determine the platform specific library path
	static {
		String os = System.getProperty("os.name").toLowerCase();

		if (os.contains("win")) {
			// Windows library name: imgal_c.dll
			libPrefix = "";
			libExtension = "dll";
		} else if (os.contains("mac") || os.contains("darwin")) {
			// macOS library name: libimgal_c.dylib
			libPrefix = "lib";
			libExtension = "dylib";
		} else {
			// linux / unix library name: libimgal_c.so
			libPrefix = "lib";
			libExtension = "so";
		}

		// contruct the platform specific library path
		libPath = "/native/" + libPrefix + libName + "." + libExtension;
		fileName = libPrefix + libName + "." + libExtension;
	}

	// copy the imgal library from resources and then load it (for SymbolLookup)
	static {
		try {
			URL url = AbstractNativeLibrary.class.getResource(libPath);
			if (url == null) {
				throw new RuntimeException("Native library " + fileName + " not found at: " + libPath);
			}
			Path tmpLib = Files.createTempFile(libPrefix + libName, libExtension);
			try (InputStream is = url.openStream()) {
				Files.copy(is, tmpLib, StandardCopyOption.REPLACE_EXISTING);
			}
			tmpLib.toFile().deleteOnExit();
			libLookup = SymbolLookup.libraryLookup(tmpLib, Arena.global());
		} catch (Exception e) {
			throw new RuntimeException("Failed to load library: " + libPath, e);
		}
	}
}
