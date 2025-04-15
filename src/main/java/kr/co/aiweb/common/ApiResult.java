/**
 * 
 */
package kr.co.aiweb.common;

import kr.co.aiweb.machinelearning.vo.LabelInfo;
import kr.co.aiweb.machinelearning.vo.PredictResult;
import lombok.Getter;
import lombok.Setter;

/**
 * 
 */
public class ApiResult {
	/**
	 */
	@Getter
	private String resultCode;
	
	/**
	 */
	@Getter
	private String resultMsg;
	
	@Getter @Setter
	private LabelInfo labelInfo;
	
	@Getter @Setter
	private PredictResult predictResult;
	
	/**
	 * @param resultCode
	 * @param resultMsg
	 */
	public void setResultCode(String resultCode, String resultMsg) {
		this.resultCode = resultCode;
		this.resultMsg = resultMsg;
	}
}
