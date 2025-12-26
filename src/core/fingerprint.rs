//! Fingerprint-based session detection for context-cache
//!
//! When `force-cache` is enabled alongside `context-cache`, this module allows
//! automatic session detection through message fingerprinting. This enables
//! transparent context reuse without requiring clients to provide session_id.

use std::collections::HashMap;

/// Number of characters to use for fingerprint prefix/suffix
const FINGERPRINT_CHAR_LEN: usize = 32;

/// Fingerprint for a single message (role + text characteristics)
#[derive(Clone, Debug, Hash, PartialEq, Eq)]
pub struct MessageFingerprint {
    pub role: String,
    pub text_len: usize,
    pub first_chars: String,
    pub last_chars: String,
}

impl MessageFingerprint {
    /// Create fingerprint from role and content text
    pub fn from_text(role: &str, content: &str) -> Self {
        let text_len = content.len();
        let first_chars: String = content.chars().take(FINGERPRINT_CHAR_LEN).collect();
        let last_chars: String = content
            .chars()
            .rev()
            .take(FINGERPRINT_CHAR_LEN)
            .collect::<String>()
            .chars()
            .rev()
            .collect();
        Self {
            role: role.to_string(),
            text_len,
            first_chars,
            last_chars,
        }
    }
}

/// Fingerprint vector for a conversation
pub type FingerprintVec = Vec<MessageFingerprint>;

/// Manager for fingerprint-based session detection
pub struct FingerprintManager {
    /// Map from fingerprint vector to session_id
    fingerprint_to_session: HashMap<FingerprintVec, String>,
    /// Reverse map: session_id to fingerprint (for updates)
    session_to_fingerprint: HashMap<String, FingerprintVec>,
    /// Pending response fingerprints: session_id -> (first_chars, last_chars)
    /// Used to match assistant messages in subsequent requests
    pending_responses: HashMap<String, (String, String)>,
    /// Counter for generating unique session IDs
    next_session_id: usize,
}

impl Default for FingerprintManager {
    fn default() -> Self {
        Self::new()
    }
}

impl FingerprintManager {
    pub fn new() -> Self {
        Self {
            fingerprint_to_session: HashMap::new(),
            session_to_fingerprint: HashMap::new(),
            pending_responses: HashMap::new(),
            next_session_id: 0,
        }
    }

    /// Compute fingerprint from messages (as role, content pairs)
    /// - If last message is user AND there's at least one assistant message: exclude last user message
    /// - Otherwise: include all messages (first request scenario)
    pub fn compute_fingerprint(messages: &[(String, String)]) -> FingerprintVec {
        if messages.is_empty() {
            return vec![];
        }

        let has_assistant = messages.iter().any(|(role, _)| role == "assistant");
        let last_is_user = messages
            .last()
            .map(|(role, _)| role == "user")
            .unwrap_or(false);

        let msgs_to_fingerprint = if has_assistant && last_is_user {
            &messages[..messages.len() - 1]
        } else {
            messages
        };

        msgs_to_fingerprint
            .iter()
            .map(|(role, content)| MessageFingerprint::from_text(role, content))
            .collect()
    }

    /// Try to find a matching session_id for the given fingerprint
    pub fn find_session(&self, fingerprint: &FingerprintVec) -> Option<String> {
        self.fingerprint_to_session.get(fingerprint).cloned()
    }

    /// Register a new fingerprint with a generated session_id
    pub fn register_session(&mut self, fingerprint: FingerprintVec) -> String {
        let session_id = format!("auto_session_{}", self.next_session_id);
        self.next_session_id += 1;
        self.fingerprint_to_session
            .insert(fingerprint.clone(), session_id.clone());
        self.session_to_fingerprint
            .insert(session_id.clone(), fingerprint);
        crate::log_info!("FingerprintManager: registered new session {}", session_id);
        session_id
    }

    /// Track response text for a session (used to match assistant messages later)
    pub fn track_response(&mut self, session_id: &str, response_text: &str) {
        let first_chars: String = response_text.chars().take(FINGERPRINT_CHAR_LEN).collect();
        let last_chars: String = response_text
            .chars()
            .rev()
            .take(FINGERPRINT_CHAR_LEN)
            .collect::<String>()
            .chars()
            .rev()
            .collect();
        self.pending_responses
            .insert(session_id.to_string(), (first_chars, last_chars));
    }

    /// Update the fingerprint for a session after response is complete
    /// This extends the fingerprint to include the new assistant message
    pub fn extend_session_fingerprint(&mut self, session_id: &str, assistant_content: &str) {
        if let Some(old_fp) = self.session_to_fingerprint.remove(session_id) {
            // Remove old mapping
            self.fingerprint_to_session.remove(&old_fp);

            // Create extended fingerprint with the assistant response
            let mut new_fp = old_fp;
            new_fp.push(MessageFingerprint::from_text(
                "assistant",
                assistant_content,
            ));

            // Store new mapping
            self.fingerprint_to_session
                .insert(new_fp.clone(), session_id.to_string());
            self.session_to_fingerprint
                .insert(session_id.to_string(), new_fp);

            crate::log_info!(
                "FingerprintManager: extended session {} fingerprint with assistant response",
                session_id
            );
        }
        // Clear pending response since it's now part of the fingerprint
        self.pending_responses.remove(session_id);
    }

    /// Remove a session's fingerprint (when cache is evicted)
    pub fn remove_session(&mut self, session_id: &str) {
        if let Some(fp) = self.session_to_fingerprint.remove(session_id) {
            self.fingerprint_to_session.remove(&fp);
            crate::log_info!("FingerprintManager: removed session {}", session_id);
        }
        self.pending_responses.remove(session_id);
    }

    /// Find or create session for given fingerprint
    /// When matching, also updates the stored fingerprint to include new messages
    pub fn find_or_create_session(
        &mut self,
        fingerprint_for_match: FingerprintVec,
        full_messages: &[(String, String)],
    ) -> String {
        if let Some(sid) = self.find_session(&fingerprint_for_match) {
            crate::log_info!("FingerprintManager: matched existing session {}", sid);
            // Update stored fingerprint to include ALL current messages (including last user)
            // This ensures the next turn can match after we extend with assistant response
            self.update_session_fingerprint(&sid, full_messages);
            sid
        } else {
            // For new sessions, store all messages as fingerprint
            let full_fingerprint: FingerprintVec = full_messages
                .iter()
                .map(|(role, content)| MessageFingerprint::from_text(role, content))
                .collect();
            self.register_session(full_fingerprint)
        }
    }

    /// Update a session's fingerprint with new messages (used when session is matched)
    fn update_session_fingerprint(&mut self, session_id: &str, messages: &[(String, String)]) {
        if let Some(old_fp) = self.session_to_fingerprint.remove(session_id) {
            // Remove old mapping
            self.fingerprint_to_session.remove(&old_fp);

            // Create full fingerprint from current messages
            let new_fp: FingerprintVec = messages
                .iter()
                .map(|(role, content)| MessageFingerprint::from_text(role, content))
                .collect();

            // Store new mapping
            self.fingerprint_to_session
                .insert(new_fp.clone(), session_id.to_string());
            self.session_to_fingerprint
                .insert(session_id.to_string(), new_fp);

            crate::log_info!(
                "FingerprintManager: updated session {} fingerprint with {} messages",
                session_id,
                messages.len()
            );
        }
    }

    /// Get the number of tracked sessions
    pub fn session_count(&self) -> usize {
        self.session_to_fingerprint.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_message_fingerprint() {
        let fp = MessageFingerprint::from_text("user", "Hello, how are you?");
        assert_eq!(fp.role, "user");
        assert_eq!(fp.text_len, 19);
        assert_eq!(fp.first_chars, "Hello, how are you?");
        assert_eq!(fp.last_chars, "Hello, how are you?");
    }

    #[test]
    fn test_long_message_fingerprint() {
        let long_content = "a".repeat(100);
        let fp = MessageFingerprint::from_text("assistant", &long_content);
        assert_eq!(fp.text_len, 100);
        assert_eq!(fp.first_chars.len(), FINGERPRINT_CHAR_LEN);
        assert_eq!(fp.last_chars.len(), FINGERPRINT_CHAR_LEN);
    }

    #[test]
    fn test_compute_fingerprint_first_request() {
        // First request: system + user, no assistant
        let messages = vec![
            ("system".to_string(), "You are helpful.".to_string()),
            ("user".to_string(), "Hello".to_string()),
        ];
        let fp = FingerprintManager::compute_fingerprint(&messages);
        // Should include all messages (no assistant to trigger exclusion)
        assert_eq!(fp.len(), 2);
    }

    #[test]
    fn test_compute_fingerprint_multi_turn() {
        // Multi-turn: has assistant, last is user
        let messages = vec![
            ("system".to_string(), "You are helpful.".to_string()),
            ("user".to_string(), "Hello".to_string()),
            ("assistant".to_string(), "Hi there!".to_string()),
            ("user".to_string(), "How are you?".to_string()),
        ];
        let fp = FingerprintManager::compute_fingerprint(&messages);
        // Should exclude last user message
        assert_eq!(fp.len(), 3);
        assert_eq!(fp[2].role, "assistant");
    }

    #[test]
    fn test_session_lifecycle() {
        let mut manager = FingerprintManager::new();

        // First request creates session
        let messages1 = vec![
            ("system".to_string(), "You are helpful.".to_string()),
            ("user".to_string(), "Hello".to_string()),
        ];
        let fp1 = FingerprintManager::compute_fingerprint(&messages1);
        let session_id = manager.find_or_create_session(fp1.clone(), &messages1);
        assert!(session_id.starts_with("auto_session_"));

        // Same fingerprint should return same session
        let session_id2 = manager.find_or_create_session(fp1, &messages1);
        assert_eq!(session_id, session_id2);

        // Extend with assistant response
        manager.extend_session_fingerprint(&session_id, "Hi there!");

        // Next turn (turn 2): should still match
        let messages2 = vec![
            ("system".to_string(), "You are helpful.".to_string()),
            ("user".to_string(), "Hello".to_string()),
            ("assistant".to_string(), "Hi there!".to_string()),
            ("user".to_string(), "How are you?".to_string()),
        ];
        let fp2 = FingerprintManager::compute_fingerprint(&messages2);
        let session_id3 = manager.find_or_create_session(fp2, &messages2);
        assert_eq!(session_id, session_id3);

        // Extend with turn 2 response
        manager.extend_session_fingerprint(&session_id, "I'm doing great!");

        // Turn 3: should still match
        let messages3 = vec![
            ("system".to_string(), "You are helpful.".to_string()),
            ("user".to_string(), "Hello".to_string()),
            ("assistant".to_string(), "Hi there!".to_string()),
            ("user".to_string(), "How are you?".to_string()),
            ("assistant".to_string(), "I'm doing great!".to_string()),
            ("user".to_string(), "What's the weather?".to_string()),
        ];
        let fp3 = FingerprintManager::compute_fingerprint(&messages3);
        let session_id4 = manager.find_or_create_session(fp3, &messages3);
        assert_eq!(session_id, session_id4);
    }
}
