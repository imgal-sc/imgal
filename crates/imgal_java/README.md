Imgal Java Bindings
===

To build the Java bindings for `imgal` use:

```bash
$ mvn clean package
```

If a test `main` function is available execuate the `.class` or `.jar` file to test it. Here's an example
testing the `NativeSum` Rust function from Java.

From the package root using `.class`:

```bash
$ java -cp bindings/java/target/classes org.imgal.statistic.NativeSum
```

From the package root using `.jar`:

```bash
$ java -cp bindings/java/target/imgal-1.0-SNAPSHOT.jar org.imgal.statistic.NativeSum
```
