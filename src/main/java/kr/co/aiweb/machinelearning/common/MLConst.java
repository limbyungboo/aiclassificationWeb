package kr.co.aiweb.machinelearning.common;

import lombok.Getter;

/**
 * machine learning const
 */
public interface MLConst {

	/**
	 * Dataset Const
	 */
	public enum DatasetConst {
		BATCHSIZE(16)
		,SEED(123)
		,WIDTH(224)
		,HEIGHT(224)
		,CHANNELS(3)
		,EPOCHS(20)
		;
		
		@Getter
		private int value;
		
		private DatasetConst(int value) {
			this.value = value;
		}
	}

}
