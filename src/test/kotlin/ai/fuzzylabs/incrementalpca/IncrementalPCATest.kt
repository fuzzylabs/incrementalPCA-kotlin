package ai.fuzzylabs.incrementalpca

import koma.create
import koma.end
import koma.mat
import org.junit.jupiter.api.BeforeAll
import org.junit.jupiter.api.Test
import java.util.*
import kotlin.math.*

class IncrementalPCATest {
    @Test
    fun testInitialisePCA() {
        val ipca = IncrementalPCA(2,2)
        ipca.initialize(testDistribution)

        assert(ipca.eigenvectors.numCols() == 2)
        assert(ipca.eigenvectors.numRows() == 2)
        val vecs = ipca.eigenvectors.to2DArray()
        val eig0 = vecs.first()
        val eig1 = vecs.last()
        assert((atan(eig0.last() / eig0.first()) - positiveSlope).absoluteValue <= tolerance)
        assert((atan(eig1.last() / eig1.first()) - negativeSlope).absoluteValue <= tolerance)
    }

    @Test
    fun testUpdatePCA() {
        val ipca = IncrementalPCA(2,2)
        ipca.initialize(testDistribution.take(initializationSize).toTypedArray())
        testDistribution.takeLast(populationSize - initializationSize).forEach { ipca.update(it) }

        assert(ipca.eigenvectors.numCols() == 2)
        assert(ipca.eigenvectors.numRows() == 2)
        val vecs = ipca.eigenvectors.to2DArray()
        val eig0 = vecs.first()
        val eig1 = vecs.last()
        assert((atan(eig0.last() / eig0.first()) - positiveSlope).absoluteValue <= tolerance)
        assert((atan(eig1.last() / eig1.first()) - negativeSlope).absoluteValue <= tolerance)
    }

    companion object {
        lateinit var testDistribution: Array<DoubleArray>
        const val tolerance: Double = 0.1 // radians, or ~5.73 degrees
        const val positiveSlope: Double = PI / 4.0
        const val negativeSlope: Double = -PI / 4.0
        const val initializationSize = 100
        const val populationSize = 5000

        @BeforeAll
        @JvmStatic
        fun setupTestDistribution() {
            val sin45 = sqrt(2.0) / 2.0
            val rot = mat[
                    sin45, -sin45 end
                            sin45, sin45
            ]

            val rng = Random(1)
            fun normalPoint(stdx: Double, stdy: Double): DoubleArray =
                doubleArrayOf(rng.nextGaussian() * stdx, rng.nextGaussian() * stdy)

            val m0 = create(Array(populationSize) { normalPoint(10.0, 1.0) }).T
            val m = (rot * m0).T

            testDistribution = m.to2DArray()
        }
    }
}