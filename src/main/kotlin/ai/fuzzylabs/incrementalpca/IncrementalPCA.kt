package ai.fuzzylabs.incrementalpca

import koma.create
import koma.extensions.map
import koma.matrix.Matrix
import koma.matrix.ejml.EJMLMatrix
import koma.util.validation.validate
import org.ejml.simple.SimpleMatrix

/**
 * Class that performs Incremental PCA
 *
 * @param[d] Dimensionality of the input
 * @param[q] Dimensionality of the PCA output
 */
class IncrementalPCA(private val d: Int, private val q: Int) {
    private var mean: Matrix<Double>? = null
    private var covarianceMatrix: Matrix<Double>? = null
    private var eigenvectors: Matrix<Double>? = null
    private var n: Int = 0

    private fun updateMean(x: Matrix<Double>) {
        if (this.mean == null) {
           this.mean = x
        } else {
            this.mean = this.mean!! + (x - this.mean!!) * 1.0 / (n.toDouble() + 1.0)
        }
    }

    private fun updateEigen() {
        val eigenDecomposition = (covarianceMatrix as EJMLMatrix).storage.eig()
        val eigenvalues = eigenDecomposition.eigenvalues.map { it.real }
        val sortedIndex = eigenvalues.zip(eigenvalues.indices).sortedByDescending { it.first }.unzip().second
        val eigenvectorsList = sortedIndex.take(q).map { eigenDecomposition.getEigenVector(it) }
        eigenvectors = EJMLMatrix(eigenvectorsList.first().concatColumns(*eigenvectorsList.takeLast(q - 1).toTypedArray()))
        covarianceMatrix?.validate { 'q' x 'd' }
    }

    fun initialise(X: Array<DoubleArray>): Array<DoubleArray> {
        // Create and check matrix
        val xMat = create(X)
        if (xMat.numCols() != d) throw Exception("Input matrix has wrong dimensions")

        // Centre data
        this.mean = xMat.mapCols { create(arrayOf(it.mean()).toDoubleArray()) }
        val xCentered = xMat.mapRows { it - mean!! }

        n = xCentered.numRows()

        // Calculate covariance
        covarianceMatrix = xCentered.T * xCentered / (n - 1)
        covarianceMatrix?.validate { 'd' x 'd' }

        // Eigen decomposition
        updateEigen()

        // Project
        return (xCentered * eigenvectors!!).to2DArray()
    }

    fun update(x: DoubleArray): DoubleArray {
        val xMat = create(x)
        if (n == 0 || covarianceMatrix == null) throw Exception("Incremental PCA was not initialised")
        if (xMat.numCols() != d) throw Exception("Input vector has wrong dimensions")

        n++
        updateMean(xMat)
        val xCentered = xMat.mapRows { it - mean!! }
        covarianceMatrix = covarianceMatrix!! * (((n - 1) / n).toDouble()) + (xCentered.T * xCentered) / (n * n)

        updateEigen()

        return (xCentered * eigenvectors!!).to2DArray().first()
    }
}