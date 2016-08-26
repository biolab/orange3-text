import keyring

SERVICE_NAME = 'Orange3 Text - {}'


class CredentialManager:
    def __init__(self, username):
        """
        Class for storing passwords in the system keyring service.

        Args:
            username: username used for storing a matching password.
        """
        self.username = username
        self.service_name = SERVICE_NAME.format(self.username)

    @property
    def key(self):
        return keyring.get_password(self.service_name, self.username)

    @key.setter
    def key(self, value):
        keyring.set_password(self.service_name, self.username, value)

    def delete_password(self):
        keyring.delete_password(self.service_name, self.username)