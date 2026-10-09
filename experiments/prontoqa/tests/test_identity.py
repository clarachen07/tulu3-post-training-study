import json
import platform
import tempfile
import unittest
from pathlib import Path
from tulu_id.common import atomic_json, frozen_identity, sha256_file, validate_frozen


class FrozenIdentityTests(unittest.TestCase):
    def test_digest_files_and_environment_must_agree(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d); (root / 'sample.txt').write_text('original')
            files = {'sample.txt': sha256_file(root / 'sample.txt')}
            env = {'python': platform.python_version(), 'packages': {}}
            data = {'files': files, 'environment': env, 'protocol_digest': frozen_identity(files, env)}
            target = root / 'manifests/frozen_protocol.json'
            atomic_json(target, data)
            validate_frozen(root)
            data['files']['sample.txt'] = 'altered-record'
            atomic_json(target, data)
            with self.assertRaisesRegex(RuntimeError, 'digest'): validate_frozen(root)
            data['files']['sample.txt'] = sha256_file(root / 'sample.txt')
            atomic_json(target, data)
            (root / 'sample.txt').write_text('modified file')
            with self.assertRaisesRegex(RuntimeError, 'Frozen input/code changed'): validate_frozen(root)


if __name__ == '__main__': unittest.main()
