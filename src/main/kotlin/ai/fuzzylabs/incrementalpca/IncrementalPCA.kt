package ai.fuzzylabs.incrementalpca

import koma.create
import koma.matrix.Matrix
import koma.matrix.ejml.EJMLMatrix
import koma.util.validation.validate
import koma.zeros
import org.ejml.simple.SimpleMatrix
import java.lang.IllegalArgumentException

/**
 * Flatten SimpleMatrix into a DoubleArray
 */
fun SimpleMatrix.toDoubleArray(): DoubleArray =
        (0 until numElements).map {
            index -> this.get(index)
        }.toDoubleArray()


/**
 * Class that performs Incremental PCA
 *
 * @param[d] Dimensionality of the input
 * @param[q] Dimensionality of the PCA output
 */
class IncrementalPCA(private val d: Int, private val q: Int) {
    private var mean: Matrix<Double> = zeros(1,d)
    private var covarianceMatrix: Matrix<Double> = zeros(d,d)

    /**
     * Eigenvectors corresponding to principle components
     */
    var eigenvectors: Matrix<Double> = zeros(d,q)
        private set
    private var n: Int = 0

    private fun updateMean(x: Matrix<Double>) {
        this.mean = this.mean + (x - this.mean) / (n.toDouble() + 1)
    }

    private fun updateEigen() {
        val eigenDecomposition = (covarianceMatrix as EJMLMatrix).storage.eig()
        val eigenvalues = eigenDecomposition.eigenvalues.map { it.real }
        val sortedIndex = eigenvalues.zip(eigenvalues.indices).sortedByDescending { it.first }.unzip().second

        val eigenvectorsArray = sortedIndex.take(q)
                .map { eigenDecomposition.getEigenVector(it).toDoubleArray() }
                .toTypedArray()
        eigenvectors = create(eigenvectorsArray).T
        eigenvectors.validate { 'q' x 'd' }
    }

    /**
     * Initialises PCA with the provided array of data points
     *
     * @param[X] Array of data points. Each data point must be of size [d]
     * @return 2D array of PCA transformed data points
     * @throws IllegalArgumentException if data points are not of size [d]
     */
    fun initialize(X: Array<DoubleArray>): Array<DoubleArray> {
        // Create and check matrix
        val xMat = create(X)
        if (xMat.numCols() != d) throw IllegalArgumentException("Input matrix has wrong dimensions")

        // Centre data
        this.mean = xMat.mapCols { create(arrayOf(it.mean()).toDoubleArray()) }
        val xCentered = xMat.mapRows { it - mean }

        n = xCentered.numRows()

        // Calculate covariance
        covarianceMatrix = xCentered.T * xCentered / (n - 1)
        covarianceMatrix.validate { 'd' x 'd' }

        // Eigen decomposition
        updateEigen()

        // Project
        return (xCentered * eigenvectors).to2DArray()
    }

    /**
     * Update PCA with a newly available data point
     *
     * @param[x] A data point to be updated with. Must have a size of [d]
     * @return An array of PCA transformed values
     * @throws Exception if PCA was not initialised
     * @throws IllegalArgumentException if an input array is not of size of [d]
     */
    fun update(x: DoubleArray): DoubleArray {
        val xMat = create(x)
        if (n == 0) throw Exception("Incremental PCA was not initialised")
        if (xMat.numCols() != d) throw IllegalArgumentException("Input vector has wrong dimensions")


        updateMean(xMat)
        val xCentered = xMat.mapRows { it - mean }
        covarianceMatrix = covarianceMatrix * ((n - 1) / n.toDouble()) + (xCentered.T * xCentered) / (n * n)

        updateEigen()

        n++

        return (xCentered * eigenvectors).to2DArray().first()
    }
}