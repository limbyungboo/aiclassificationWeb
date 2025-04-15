/**
 * 
 */
package kr.co.aiweb.machinelearning.vo;

import org.nd4j.linalg.api.ndarray.INDArray;

/**
 * 테스트 결과 vo
 */
public class PredictResult {

	private INDArray output;
	
	private LabelInfo labeInfo;
	
	/**constructor
	 * @param labeInfo
	 * @param output
	 */
	public PredictResult(LabelInfo labeInfo, INDArray output) {
		this.labeInfo = labeInfo;
		this.output = output;
	}
	
	/**label index
	 * @return
	 */
	public int getPredictedIndex() {
		return output.argMax(1).getInt(0);
	}
	
	/**confidence
	 * @return
	 */
	public double getConfidence() {
		return output.getDouble(0, getPredictedIndex());
	}
	
	/**label 명칭
	 * @return
	 */
	public String getLabelName() {
		return labeInfo.getLabelName(getPredictedIndex());
	}
	
}
