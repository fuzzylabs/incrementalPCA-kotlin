# Incremental PCA
[![](https://jitpack.io/v/fuzzylabs/incrementalPCA-kotlin.svg)](https://jitpack.io/#fuzzylabs/incrementalPCA-kotlin)

The library provides methods for online PCA, based on incremental updates of the mean vector and the covariance matrix of real-time data, and eigen-value decomposition on the resulting covariance matrices.

## Dependencies
* [Koma](http://koma.kyonifer.com/index.html) for matrix operations
* [JUnit 5.7](https://junit.org/junit5/) for unit tests

## Build

Assuming Gradle is in PATH.

To run unit tests:
```
gradle test
```

To build class files:

```
gradle build
```

To build and create .jar file

```
gradle jar
```

## Get from JitPack

Using Gradle, add JitPack as a repository

```
	allprojects {
		repositories {
			...
			maven { url 'https://jitpack.io' }
		}
	}
```

and specify the library as a dependency
```
	dependencies {
	        implementation 'com.github.fuzzylabs:incrementalPCA-kotlin:master-SNAPSHOT'
	}
```

If you are not using Gradle, for other options, visit [package repository](https://jitpack.io/#fuzzylabs/incrementalPCA-kotlin).

## Usage

Initialise PCA with some initial data

```
val ipca = IncrementalPCA(d = 3, q = 1)
val projection = ipca.initialise(data) 
// Returns Array<DoubleArray> of projected data points
```

Update PCA when a new data point is available

```
val projection = ipca.update(data) 
// Returns DoubleArray of a projected data point
```

## Motivation

For [Wearable My Foot project](https://github.com/fuzzylabs/wearable-my-foot/), we need to preprocess the incoming signal, in such a way that we get a primary direction of motion. Analyzing data in a notebook after it has been collected, it is possible to perform PCA on the whole series of data. However, for the purpose of analysis on an Android device in real-time we need to update PCA in real-time as well (i.e. use online PCA algorithm).
 
To the best of our knowledge, there's no suitable implementation of online PCA algorithms that can be used on Android. Therefore, we implemented it ourself.

## Limitations

As the algorithm relies on the eigen-value decomposition, which has time complexity of *O(n^3)*, it may not be suitable for problems with many features. In such case, usage of inexact SVD that is more efficient may be required.

## References
* [Stackoverflow answer](https://stats.stackexchange.com/a/451923) with the simplified formula for the covariance and mean vector updates 
* [Sanjoy Dasgupta and Daniel Hsu, "On-line estimation with the multivariate Gaussian distribution"](https://www.cs.columbia.edu/~djhsu/papers/gauss.pdf) the paper that describes the validity of the above formulas in more details
