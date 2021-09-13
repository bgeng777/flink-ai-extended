# Generated by the gRPC Python protocol compiler plugin. DO NOT EDIT!
"""Client and server classes corresponding to protobuf-defined services."""
import grpc

from . import notification_service_pb2 as notification__service__pb2


class NotificationServiceStub(object):
    """AirFlowService provides notification function rest endpoint of NotificationService for Notification Service component.
    Functions of NotificationService include:
    1.Send event.
    2.List events.
    """

    def __init__(self, channel):
        """Constructor.

        Args:
            channel: A grpc.Channel.
        """
        self.sendEvent = channel.unary_unary(
                '/notification_service.NotificationService/sendEvent',
                request_serializer=notification__service__pb2.SendEventRequest.SerializeToString,
                response_deserializer=notification__service__pb2.SendEventsResponse.FromString,
                )
        self.listEvents = channel.unary_unary(
                '/notification_service.NotificationService/listEvents',
                request_serializer=notification__service__pb2.ListEventsRequest.SerializeToString,
                response_deserializer=notification__service__pb2.ListEventsResponse.FromString,
                )
        self.listAllEvents = channel.unary_unary(
                '/notification_service.NotificationService/listAllEvents',
                request_serializer=notification__service__pb2.ListAllEventsRequest.SerializeToString,
                response_deserializer=notification__service__pb2.ListEventsResponse.FromString,
                )
        self.notify = channel.unary_unary(
                '/notification_service.NotificationService/notify',
                request_serializer=notification__service__pb2.NotifyRequest.SerializeToString,
                response_deserializer=notification__service__pb2.NotifyResponse.FromString,
                )
        self.listMembers = channel.unary_unary(
                '/notification_service.NotificationService/listMembers',
                request_serializer=notification__service__pb2.ListMembersRequest.SerializeToString,
                response_deserializer=notification__service__pb2.ListMembersResponse.FromString,
                )
        self.notifyNewMember = channel.unary_unary(
                '/notification_service.NotificationService/notifyNewMember',
                request_serializer=notification__service__pb2.NotifyNewMemberRequest.SerializeToString,
                response_deserializer=notification__service__pb2.NotifyNewMemberResponse.FromString,
                )
        self.getLatestVersionByKey = channel.unary_unary(
                '/notification_service.NotificationService/getLatestVersionByKey',
                request_serializer=notification__service__pb2.GetLatestVersionByKeyRequest.SerializeToString,
                response_deserializer=notification__service__pb2.GetLatestVersionResponse.FromString,
                )
        self.registerClient = channel.unary_unary(
                '/notification_service.NotificationService/registerClient',
                request_serializer=notification__service__pb2.RegisterClientRequest.SerializeToString,
                response_deserializer=notification__service__pb2.RegisterClientResponse.FromString,
                )


class NotificationServiceServicer(object):
    """AirFlowService provides notification function rest endpoint of NotificationService for Notification Service component.
    Functions of NotificationService include:
    1.Send event.
    2.List events.
    """

    def sendEvent(self, request, context):
        """Send event.
        """
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details('Method not implemented!')
        raise NotImplementedError('Method not implemented!')

    def listEvents(self, request, context):
        """List events.
        """
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details('Method not implemented!')
        raise NotImplementedError('Method not implemented!')

    def listAllEvents(self, request, context):
        """List all events
        """
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details('Method not implemented!')
        raise NotImplementedError('Method not implemented!')

    def notify(self, request, context):
        """Accepts notifications from other members.
        """
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details('Method not implemented!')
        raise NotImplementedError('Method not implemented!')

    def listMembers(self, request, context):
        """List current living members.
        """
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details('Method not implemented!')
        raise NotImplementedError('Method not implemented!')

    def notifyNewMember(self, request, context):
        """Notify current members that there is a new member added.
        """
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details('Method not implemented!')
        raise NotImplementedError('Method not implemented!')

    def getLatestVersionByKey(self, request, context):
        """Get latest version by key
        """
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details('Method not implemented!')
        raise NotImplementedError('Method not implemented!')

    def registerClient(self, request, context):
        """Register notification client in the db of notification service
        """
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details('Method not implemented!')
        raise NotImplementedError('Method not implemented!')


def add_NotificationServiceServicer_to_server(servicer, server):
    rpc_method_handlers = {
            'sendEvent': grpc.unary_unary_rpc_method_handler(
                    servicer.sendEvent,
                    request_deserializer=notification__service__pb2.SendEventRequest.FromString,
                    response_serializer=notification__service__pb2.SendEventsResponse.SerializeToString,
            ),
            'listEvents': grpc.unary_unary_rpc_method_handler(
                    servicer.listEvents,
                    request_deserializer=notification__service__pb2.ListEventsRequest.FromString,
                    response_serializer=notification__service__pb2.ListEventsResponse.SerializeToString,
            ),
            'listAllEvents': grpc.unary_unary_rpc_method_handler(
                    servicer.listAllEvents,
                    request_deserializer=notification__service__pb2.ListAllEventsRequest.FromString,
                    response_serializer=notification__service__pb2.ListEventsResponse.SerializeToString,
            ),
            'notify': grpc.unary_unary_rpc_method_handler(
                    servicer.notify,
                    request_deserializer=notification__service__pb2.NotifyRequest.FromString,
                    response_serializer=notification__service__pb2.NotifyResponse.SerializeToString,
            ),
            'listMembers': grpc.unary_unary_rpc_method_handler(
                    servicer.listMembers,
                    request_deserializer=notification__service__pb2.ListMembersRequest.FromString,
                    response_serializer=notification__service__pb2.ListMembersResponse.SerializeToString,
            ),
            'notifyNewMember': grpc.unary_unary_rpc_method_handler(
                    servicer.notifyNewMember,
                    request_deserializer=notification__service__pb2.NotifyNewMemberRequest.FromString,
                    response_serializer=notification__service__pb2.NotifyNewMemberResponse.SerializeToString,
            ),
            'getLatestVersionByKey': grpc.unary_unary_rpc_method_handler(
                    servicer.getLatestVersionByKey,
                    request_deserializer=notification__service__pb2.GetLatestVersionByKeyRequest.FromString,
                    response_serializer=notification__service__pb2.GetLatestVersionResponse.SerializeToString,
            ),
            'registerClient': grpc.unary_unary_rpc_method_handler(
                    servicer.registerClient,
                    request_deserializer=notification__service__pb2.RegisterClientRequest.FromString,
                    response_serializer=notification__service__pb2.RegisterClientResponse.SerializeToString,
            ),
    }
    generic_handler = grpc.method_handlers_generic_handler(
            'notification_service.NotificationService', rpc_method_handlers)
    server.add_generic_rpc_handlers((generic_handler,))


 # This class is part of an EXPERIMENTAL API.
class NotificationService(object):
    """AirFlowService provides notification function rest endpoint of NotificationService for Notification Service component.
    Functions of NotificationService include:
    1.Send event.
    2.List events.
    """

    @staticmethod
    def sendEvent(request,
            target,
            options=(),
            channel_credentials=None,
            call_credentials=None,
            insecure=False,
            compression=None,
            wait_for_ready=None,
            timeout=None,
            metadata=None):
        return grpc.experimental.unary_unary(request, target, '/notification_service.NotificationService/sendEvent',
            notification__service__pb2.SendEventRequest.SerializeToString,
            notification__service__pb2.SendEventsResponse.FromString,
            options, channel_credentials,
            insecure, call_credentials, compression, wait_for_ready, timeout, metadata)

    @staticmethod
    def listEvents(request,
            target,
            options=(),
            channel_credentials=None,
            call_credentials=None,
            insecure=False,
            compression=None,
            wait_for_ready=None,
            timeout=None,
            metadata=None):
        return grpc.experimental.unary_unary(request, target, '/notification_service.NotificationService/listEvents',
            notification__service__pb2.ListEventsRequest.SerializeToString,
            notification__service__pb2.ListEventsResponse.FromString,
            options, channel_credentials,
            insecure, call_credentials, compression, wait_for_ready, timeout, metadata)

    @staticmethod
    def listAllEvents(request,
            target,
            options=(),
            channel_credentials=None,
            call_credentials=None,
            insecure=False,
            compression=None,
            wait_for_ready=None,
            timeout=None,
            metadata=None):
        return grpc.experimental.unary_unary(request, target, '/notification_service.NotificationService/listAllEvents',
            notification__service__pb2.ListAllEventsRequest.SerializeToString,
            notification__service__pb2.ListEventsResponse.FromString,
            options, channel_credentials,
            insecure, call_credentials, compression, wait_for_ready, timeout, metadata)

    @staticmethod
    def notify(request,
            target,
            options=(),
            channel_credentials=None,
            call_credentials=None,
            insecure=False,
            compression=None,
            wait_for_ready=None,
            timeout=None,
            metadata=None):
        return grpc.experimental.unary_unary(request, target, '/notification_service.NotificationService/notify',
            notification__service__pb2.NotifyRequest.SerializeToString,
            notification__service__pb2.NotifyResponse.FromString,
            options, channel_credentials,
            insecure, call_credentials, compression, wait_for_ready, timeout, metadata)

    @staticmethod
    def listMembers(request,
            target,
            options=(),
            channel_credentials=None,
            call_credentials=None,
            insecure=False,
            compression=None,
            wait_for_ready=None,
            timeout=None,
            metadata=None):
        return grpc.experimental.unary_unary(request, target, '/notification_service.NotificationService/listMembers',
            notification__service__pb2.ListMembersRequest.SerializeToString,
            notification__service__pb2.ListMembersResponse.FromString,
            options, channel_credentials,
            insecure, call_credentials, compression, wait_for_ready, timeout, metadata)

    @staticmethod
    def notifyNewMember(request,
            target,
            options=(),
            channel_credentials=None,
            call_credentials=None,
            insecure=False,
            compression=None,
            wait_for_ready=None,
            timeout=None,
            metadata=None):
        return grpc.experimental.unary_unary(request, target, '/notification_service.NotificationService/notifyNewMember',
            notification__service__pb2.NotifyNewMemberRequest.SerializeToString,
            notification__service__pb2.NotifyNewMemberResponse.FromString,
            options, channel_credentials,
            insecure, call_credentials, compression, wait_for_ready, timeout, metadata)

    @staticmethod
    def getLatestVersionByKey(request,
            target,
            options=(),
            channel_credentials=None,
            call_credentials=None,
            insecure=False,
            compression=None,
            wait_for_ready=None,
            timeout=None,
            metadata=None):
        return grpc.experimental.unary_unary(request, target, '/notification_service.NotificationService/getLatestVersionByKey',
            notification__service__pb2.GetLatestVersionByKeyRequest.SerializeToString,
            notification__service__pb2.GetLatestVersionResponse.FromString,
            options, channel_credentials,
            insecure, call_credentials, compression, wait_for_ready, timeout, metadata)

    @staticmethod
    def registerClient(request,
            target,
            options=(),
            channel_credentials=None,
            call_credentials=None,
            insecure=False,
            compression=None,
            wait_for_ready=None,
            timeout=None,
            metadata=None):
        return grpc.experimental.unary_unary(request, target, '/notification_service.NotificationService/registerClient',
            notification__service__pb2.RegisterClientRequest.SerializeToString,
            notification__service__pb2.RegisterClientResponse.FromString,
            options, channel_credentials,
            insecure, call_credentials, compression, wait_for_ready, timeout, metadata)
