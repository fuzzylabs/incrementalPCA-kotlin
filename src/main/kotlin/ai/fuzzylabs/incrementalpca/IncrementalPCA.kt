package ai.fuzzylabs.incrementalpca

import koma.create
import koma.extensions.map
import koma.matrix.Matrix
import koma.matrix.ejml.EJMLMatrix
import koma.util.validation.validate
import koma.zeros
import org.ejml.simple.SimpleMatrix

/**
 * Class that performs Incremental PCA
 *
 * @param[d] Dimensionality of the input
 * @param[q] Dimensionality of the PCA output
 */
class IncrementalPCA(private val d: Int, private val q: Int) {
    private var mean: Matrix<Double> = zeros(1,d)
    private var covarianceMatrix: Matrix<Double> = zeros(d,d)
    var eigenvectors: Matrix<Double> = zeros(d,q)
        private set
    private var n: Int = 0

    private fun updateMean(x: Matrix<Double>) {
        this.mean = this.mean + (x - this.mean) * 1.0 / (n.toDouble() + 1.0)
    }

    private fun updateEigen() {
        val eigenDecomposition = (covarianceMatrix as EJMLMatrix).storage.eig()
        val eigenvalues = eigenDecomposition.eigenvalues.map { it.real }
        val sortedIndex = eigenvalues.zip(eigenvalues.indices).sortedByDescending { it.first }.unzip().second

        val eigenvectorsList = sortedIndex.take(q).map { eigenDecomposition.getEigenVector(it) }
        eigenvectors = EJMLMatrix(eigenvectorsList.first().concatColumns(*eigenvectorsList.takeLast(q - 1).toTypedArray()))
        covarianceMatrix.validate { 'q' x 'd' }
    }

    fun initialize(X: Array<DoubleArray>): Array<DoubleArray> {
        // Create and check matrix
        val xMat = create(X)
        if (xMat.numCols() != d) throw Exception("Input matrix has wrong dimensions")

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

    fun update(x: DoubleArray): DoubleArray {
        val xMat = create(x)
        if (n == 0) throw Exception("Incremental PCA was not initialised")
        if (xMat.numCols() != d) throw Exception("Input vector has wrong dimensions")

        n++
        updateMean(xMat)
        val xCentered = xMat.mapRows { it - mean }
        covarianceMatrix = covarianceMatrix * (((n - 1) / n).toDouble()) + (xCentered.T * xCentered) / (n * n)

        updateEigen()

        return (xCentered * eigenvectors).to2DArray().first()
    }
}