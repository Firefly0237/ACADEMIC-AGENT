"""Trusted server endpoints with existing native schema and proof validators."""
from citequest.index_v2.native import NativeMilvus
from citequest.index_v2.schema import checked_endpoint

from .backend import read_profile
from .native import GPUCandidateNative


class PublicationNative(GPUCandidateNative):
    def __init__(self, transport, *, backend_profile_file=None):
        self.gpu = transport.backend_profile_sha256 is not None
        if self.gpu:
            if backend_profile_file is None:
                raise ValueError("explicit_existing_backend_profile_file_required")
            self.backend_profile, self.backend_sha256, _ = read_profile(backend_profile_file)
            if self.backend_sha256 != transport.backend_profile_sha256 or self.backend_profile["function_endpoint"] != transport.function_endpoint:
                raise ValueError("publication_backend_runtime_binding_mismatch")
            self.profile_path, self.info_url = backend_profile_file, transport.tei_info_url
            self.endpoint = transport.function_endpoint
        else:
            self.endpoint = checked_endpoint(transport.function_endpoint)
            self.tei_info_url = transport.tei_info_url
        transport._connect()
        self.audit, self.client, self.channel = transport.audit, transport.client, transport.channel
        self.database, self.absolute_deadline = transport.database, None

    def create(self, name):
        return GPUCandidateNative.create(self, name) if self.gpu else NativeMilvus.create(self, name)

    def collection_proof(self, name):
        return GPUCandidateNative.collection_proof(self, name) if self.gpu else NativeMilvus.collection_proof(self, name)
