/**
 * 
 */
package kr.co.aiweb.config;

import java.io.IOException;
import java.util.List;
import java.util.concurrent.CopyOnWriteArrayList;

import org.springframework.stereotype.Component;
import org.springframework.web.socket.CloseStatus;
import org.springframework.web.socket.TextMessage;
import org.springframework.web.socket.WebSocketSession;
import org.springframework.web.socket.handler.TextWebSocketHandler;

import lombok.extern.slf4j.Slf4j;

/**
 * websocket handler
 */
@Slf4j
@Component
public class TrainingLogWebSocketHandler extends TextWebSocketHandler {
	
    /**
     * sessions
     */
    private static final List<WebSocketSession> sessions = new CopyOnWriteArrayList<>();

    /**broad cast
     * @param message
     */
    public static void broadcast(String message) {
        for (WebSocketSession session : sessions) {
            try {
            	log.info(" >>>>>>>>>>>>> broadcast");
                if (session.isOpen()) {
                	log.info(" >>>>>>>>>>>>> broadcast isOpen");
                    session.sendMessage(new TextMessage(message));
                }
                else {
                	log.info(" >>>>>>>>>>>>> broadcast isNotOpen");
                }
            } 
            catch (IOException e) {
            	log.error(e.getMessage(), e);
            }
        }
    }

    @Override
    public void afterConnectionEstablished(WebSocketSession session) {
    	log.info(" >>>>>>>>>>>>>>>> afterConnectionEstablished");
        sessions.add(session);
    }

    @Override
    public void afterConnectionClosed(WebSocketSession session, CloseStatus status) {
    	log.info(" >>>>>>>>>>>>>>>> afterConnectionClosed");
        sessions.remove(session);
    }
}
