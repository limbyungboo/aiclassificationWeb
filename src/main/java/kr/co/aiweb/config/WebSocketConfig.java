/**
 * 
 */
package kr.co.aiweb.config;

import org.springframework.context.annotation.Configuration;
import org.springframework.web.socket.config.annotation.EnableWebSocket;
import org.springframework.web.socket.config.annotation.WebSocketConfigurer;
import org.springframework.web.socket.config.annotation.WebSocketHandlerRegistry;

import lombok.extern.slf4j.Slf4j;

/**
 * 
 */
@Slf4j
@Configuration
@EnableWebSocket
public class WebSocketConfig implements WebSocketConfigurer {
	
	/**@Override 
	 * @see org.springframework.web.socket.config.annotation.WebSocketConfigurer#registerWebSocketHandlers(org.springframework.web.socket.config.annotation.WebSocketHandlerRegistry)
	 */
	@Override
    public void registerWebSocketHandlers(WebSocketHandlerRegistry registry) {
		log.info("########## WebSocketConfig registerWebSocketHandlers ----------");
        registry.addHandler(new TrainingLogWebSocketHandler(), "/ws/train_chart")
                .setAllowedOrigins("*");
    }
}
