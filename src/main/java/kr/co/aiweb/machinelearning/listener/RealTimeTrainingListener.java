/**
 * 
 */
package kr.co.aiweb.machinelearning.listener;

import org.deeplearning4j.nn.api.Model;
import org.deeplearning4j.optimize.api.BaseTrainingListener;

import kr.co.aiweb.config.TrainingLogWebSocketHandler;
import lombok.extern.slf4j.Slf4j;

/**
 * 
 */
@Slf4j
public class RealTimeTrainingListener extends BaseTrainingListener {
	
	/**@Override 
	 * @see org.deeplearning4j.optimize.api.BaseTrainingListener#iterationDone(org.deeplearning4j.nn.api.Model, int, int)
	 */
	@Override
	public void iterationDone(Model model, int iteration, int epoch) {
		double score = model.score();
		String message = String.format("{\"iteration\":%d, \"loss\":%.6f}", iteration, score);
		log.info("RealTimeTrainingListener iterationDone message = {}", message);
		TrainingLogWebSocketHandler.broadcast(message);
	}
}
