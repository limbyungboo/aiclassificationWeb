package kr.co.aiweb.machinelearning.common;

import lombok.Getter;

/**
 * machine learning const
 */
public interface MLConst {
	
	/**
	 * Dataset Const
	 */
	public enum MLDatasetConst {
		BATCHSIZE(16)
		,SEED(123)
		,WIDTH(224)
		,HEIGHT(224)
		,CHANNELS(3)
		,EPOCHS(10)
		;
		
		@Getter
		private final int value;
		
		private MLDatasetConst(int value) {
			this.value = value;
		}
	}
}
