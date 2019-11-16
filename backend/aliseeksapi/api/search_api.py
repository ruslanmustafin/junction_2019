# coding: utf-8

"""
    Aliseeks API

    AliExpress product searching and product details retrieval API.   # noqa: E501

    OpenAPI spec version: 1.0.0
    Generated by: https://openapi-generator.tech
"""


from __future__ import absolute_import

import re  # noqa: F401

# python 2 and python 3 compatibility library
import six

from aliseeksapi.api_client import ApiClient


class SearchApi(object):
    """NOTE: This class is auto generated by OpenAPI Generator
    Ref: https://openapi-generator.tech

    Do not edit the class manually.
    """

    def __init__(self, api_client=None):
        if api_client is None:
            api_client = ApiClient()
        self.api_client = api_client

    def realtime_search(self, realtime_search_request, **kwargs):  # noqa: E501
        """Searches AliExpress in realtime   # noqa: E501

        This method makes a synchronous HTTP request by default. To make an
        asynchronous HTTP request, please pass async_req=True
        >>> thread = api.realtime_search(realtime_search_request, async_req=True)
        >>> result = thread.get()

        :param async_req bool
        :param RealtimeSearchRequest realtime_search_request: Realtime search request body  (required)
        :return: RealtimeSearchResponse
                 If the method is called asynchronously,
                 returns the request thread.
        """
        kwargs['_return_http_data_only'] = True
        if kwargs.get('async_req'):
            return self.realtime_search_with_http_info(realtime_search_request, **kwargs)  # noqa: E501
        else:
            (data) = self.realtime_search_with_http_info(realtime_search_request, **kwargs)  # noqa: E501
            return data

    def realtime_search_with_http_info(self, realtime_search_request, **kwargs):  # noqa: E501
        """Searches AliExpress in realtime   # noqa: E501

        This method makes a synchronous HTTP request by default. To make an
        asynchronous HTTP request, please pass async_req=True
        >>> thread = api.realtime_search_with_http_info(realtime_search_request, async_req=True)
        >>> result = thread.get()

        :param async_req bool
        :param RealtimeSearchRequest realtime_search_request: Realtime search request body  (required)
        :return: RealtimeSearchResponse
                 If the method is called asynchronously,
                 returns the request thread.
        """

        local_var_params = locals()

        all_params = ['realtime_search_request']  # noqa: E501
        all_params.append('async_req')
        all_params.append('_return_http_data_only')
        all_params.append('_preload_content')
        all_params.append('_request_timeout')

        for key, val in six.iteritems(local_var_params['kwargs']):
            if key not in all_params:
                raise TypeError(
                    "Got an unexpected keyword argument '%s'"
                    " to method realtime_search" % key
                )
            local_var_params[key] = val
        del local_var_params['kwargs']
        # verify the required parameter 'realtime_search_request' is set
        if ('realtime_search_request' not in local_var_params or
                local_var_params['realtime_search_request'] is None):
            raise ValueError("Missing the required parameter `realtime_search_request` when calling `realtime_search`")  # noqa: E501

        collection_formats = {}

        path_params = {}

        query_params = []

        header_params = {}

        form_params = []
        local_var_files = {}

        body_params = None
        if 'realtime_search_request' in local_var_params:
            body_params = local_var_params['realtime_search_request']
        # HTTP header `Accept`
        header_params['Accept'] = self.api_client.select_header_accept(
            ['application/json'])  # noqa: E501

        # HTTP header `Content-Type`
        header_params['Content-Type'] = self.api_client.select_header_content_type(  # noqa: E501
            ['application/json'])  # noqa: E501

        # Authentication setting
        auth_settings = ['ApiKeyAuth']  # noqa: E501

        return self.api_client.call_api(
            '/search/realtime', 'POST',
            path_params,
            query_params,
            header_params,
            body=body_params,
            post_params=form_params,
            files=local_var_files,
            response_type='RealtimeSearchResponse',  # noqa: E501
            auth_settings=auth_settings,
            async_req=local_var_params.get('async_req'),
            _return_http_data_only=local_var_params.get('_return_http_data_only'),  # noqa: E501
            _preload_content=local_var_params.get('_preload_content', True),
            _request_timeout=local_var_params.get('_request_timeout'),
            collection_formats=collection_formats)

    def search(self, search_request, **kwargs):  # noqa: E501
        """Searches AliExpress in non-realtime. Uses the Aliseeks.com datasource which is continually updated from AliExpress.   # noqa: E501

        This method makes a synchronous HTTP request by default. To make an
        asynchronous HTTP request, please pass async_req=True
        >>> thread = api.search(search_request, async_req=True)
        >>> result = thread.get()

        :param async_req bool
        :param SearchRequest search_request: Search request body  (required)
        :return: SearchResponse
                 If the method is called asynchronously,
                 returns the request thread.
        """
        kwargs['_return_http_data_only'] = True
        if kwargs.get('async_req'):
            return self.search_with_http_info(search_request, **kwargs)  # noqa: E501
        else:
            (data) = self.search_with_http_info(search_request, **kwargs)  # noqa: E501
            return data

    def search_with_http_info(self, search_request, **kwargs):  # noqa: E501
        """Searches AliExpress in non-realtime. Uses the Aliseeks.com datasource which is continually updated from AliExpress.   # noqa: E501

        This method makes a synchronous HTTP request by default. To make an
        asynchronous HTTP request, please pass async_req=True
        >>> thread = api.search_with_http_info(search_request, async_req=True)
        >>> result = thread.get()

        :param async_req bool
        :param SearchRequest search_request: Search request body  (required)
        :return: SearchResponse
                 If the method is called asynchronously,
                 returns the request thread.
        """

        local_var_params = locals()

        all_params = ['search_request']  # noqa: E501
        all_params.append('async_req')
        all_params.append('_return_http_data_only')
        all_params.append('_preload_content')
        all_params.append('_request_timeout')

        for key, val in six.iteritems(local_var_params['kwargs']):
            if key not in all_params:
                raise TypeError(
                    "Got an unexpected keyword argument '%s'"
                    " to method search" % key
                )
            local_var_params[key] = val
        del local_var_params['kwargs']
        # verify the required parameter 'search_request' is set
        if ('search_request' not in local_var_params or
                local_var_params['search_request'] is None):
            raise ValueError("Missing the required parameter `search_request` when calling `search`")  # noqa: E501

        collection_formats = {}

        path_params = {}

        query_params = []

        header_params = {}

        form_params = []
        local_var_files = {}

        body_params = None
        if 'search_request' in local_var_params:
            body_params = local_var_params['search_request']
        # HTTP header `Accept`
        header_params['Accept'] = self.api_client.select_header_accept(
            ['application/json'])  # noqa: E501

        # HTTP header `Content-Type`
        header_params['Content-Type'] = self.api_client.select_header_content_type(  # noqa: E501
            ['application/json'])  # noqa: E501

        # Authentication setting
        auth_settings = ['ApiKeyAuth']  # noqa: E501

        return self.api_client.call_api(
            '/search', 'POST',
            path_params,
            query_params,
            header_params,
            body=body_params,
            post_params=form_params,
            files=local_var_files,
            response_type='SearchResponse',  # noqa: E501
            auth_settings=auth_settings,
            async_req=local_var_params.get('async_req'),
            _return_http_data_only=local_var_params.get('_return_http_data_only'),  # noqa: E501
            _preload_content=local_var_params.get('_preload_content', True),
            _request_timeout=local_var_params.get('_request_timeout'),
            collection_formats=collection_formats)

    def search_best_selling(self, best_selling_search_request, **kwargs):  # noqa: E501
        """Retrieves best selling products from AliExpress in realtime.   # noqa: E501

        This method makes a synchronous HTTP request by default. To make an
        asynchronous HTTP request, please pass async_req=True
        >>> thread = api.search_best_selling(best_selling_search_request, async_req=True)
        >>> result = thread.get()

        :param async_req bool
        :param BestSellingSearchRequest best_selling_search_request: Search best selling request body  (required)
        :return: BestSellingSearchResponse
                 If the method is called asynchronously,
                 returns the request thread.
        """
        kwargs['_return_http_data_only'] = True
        if kwargs.get('async_req'):
            return self.search_best_selling_with_http_info(best_selling_search_request, **kwargs)  # noqa: E501
        else:
            (data) = self.search_best_selling_with_http_info(best_selling_search_request, **kwargs)  # noqa: E501
            return data

    def search_best_selling_with_http_info(self, best_selling_search_request, **kwargs):  # noqa: E501
        """Retrieves best selling products from AliExpress in realtime.   # noqa: E501

        This method makes a synchronous HTTP request by default. To make an
        asynchronous HTTP request, please pass async_req=True
        >>> thread = api.search_best_selling_with_http_info(best_selling_search_request, async_req=True)
        >>> result = thread.get()

        :param async_req bool
        :param BestSellingSearchRequest best_selling_search_request: Search best selling request body  (required)
        :return: BestSellingSearchResponse
                 If the method is called asynchronously,
                 returns the request thread.
        """

        local_var_params = locals()

        all_params = ['best_selling_search_request']  # noqa: E501
        all_params.append('async_req')
        all_params.append('_return_http_data_only')
        all_params.append('_preload_content')
        all_params.append('_request_timeout')

        for key, val in six.iteritems(local_var_params['kwargs']):
            if key not in all_params:
                raise TypeError(
                    "Got an unexpected keyword argument '%s'"
                    " to method search_best_selling" % key
                )
            local_var_params[key] = val
        del local_var_params['kwargs']
        # verify the required parameter 'best_selling_search_request' is set
        if ('best_selling_search_request' not in local_var_params or
                local_var_params['best_selling_search_request'] is None):
            raise ValueError("Missing the required parameter `best_selling_search_request` when calling `search_best_selling`")  # noqa: E501

        collection_formats = {}

        path_params = {}

        query_params = []

        header_params = {}

        form_params = []
        local_var_files = {}

        body_params = None
        if 'best_selling_search_request' in local_var_params:
            body_params = local_var_params['best_selling_search_request']
        # HTTP header `Accept`
        header_params['Accept'] = self.api_client.select_header_accept(
            ['application/json'])  # noqa: E501

        # HTTP header `Content-Type`
        header_params['Content-Type'] = self.api_client.select_header_content_type(  # noqa: E501
            ['application/json'])  # noqa: E501

        # Authentication setting
        auth_settings = ['ApiKeyAuth']  # noqa: E501

        return self.api_client.call_api(
            '/search/bestSelling', 'POST',
            path_params,
            query_params,
            header_params,
            body=body_params,
            post_params=form_params,
            files=local_var_files,
            response_type='BestSellingSearchResponse',  # noqa: E501
            auth_settings=auth_settings,
            async_req=local_var_params.get('async_req'),
            _return_http_data_only=local_var_params.get('_return_http_data_only'),  # noqa: E501
            _preload_content=local_var_params.get('_preload_content', True),
            _request_timeout=local_var_params.get('_request_timeout'),
            collection_formats=collection_formats)

    def search_by_image(self, image_search_request, **kwargs):  # noqa: E501
        """Searches AliExpress by image in realtime.   # noqa: E501

        This method makes a synchronous HTTP request by default. To make an
        asynchronous HTTP request, please pass async_req=True
        >>> thread = api.search_by_image(image_search_request, async_req=True)
        >>> result = thread.get()

        :param async_req bool
        :param ImageSearchRequest image_search_request: The image search request body  (required)
        :return: ImageSearchResponse
                 If the method is called asynchronously,
                 returns the request thread.
        """
        kwargs['_return_http_data_only'] = True
        if kwargs.get('async_req'):
            return self.search_by_image_with_http_info(image_search_request, **kwargs)  # noqa: E501
        else:
            (data) = self.search_by_image_with_http_info(image_search_request, **kwargs)  # noqa: E501
            return data

    def search_by_image_with_http_info(self, image_search_request, **kwargs):  # noqa: E501
        """Searches AliExpress by image in realtime.   # noqa: E501

        This method makes a synchronous HTTP request by default. To make an
        asynchronous HTTP request, please pass async_req=True
        >>> thread = api.search_by_image_with_http_info(image_search_request, async_req=True)
        >>> result = thread.get()

        :param async_req bool
        :param ImageSearchRequest image_search_request: The image search request body  (required)
        :return: ImageSearchResponse
                 If the method is called asynchronously,
                 returns the request thread.
        """

        local_var_params = locals()

        all_params = ['image_search_request']  # noqa: E501
        all_params.append('async_req')
        all_params.append('_return_http_data_only')
        all_params.append('_preload_content')
        all_params.append('_request_timeout')

        for key, val in six.iteritems(local_var_params['kwargs']):
            if key not in all_params:
                raise TypeError(
                    "Got an unexpected keyword argument '%s'"
                    " to method search_by_image" % key
                )
            local_var_params[key] = val
        del local_var_params['kwargs']
        # verify the required parameter 'image_search_request' is set
        if ('image_search_request' not in local_var_params or
                local_var_params['image_search_request'] is None):
            raise ValueError("Missing the required parameter `image_search_request` when calling `search_by_image`")  # noqa: E501

        collection_formats = {}

        path_params = {}

        query_params = []

        header_params = {}

        form_params = []
        local_var_files = {}

        body_params = None
        if 'image_search_request' in local_var_params:
            body_params = local_var_params['image_search_request']
        # HTTP header `Accept`
        header_params['Accept'] = self.api_client.select_header_accept(
            ['application/json'])  # noqa: E501

        # HTTP header `Content-Type`
        header_params['Content-Type'] = self.api_client.select_header_content_type(  # noqa: E501
            ['application/json'])  # noqa: E501

        # Authentication setting
        auth_settings = ['ApiKeyAuth']  # noqa: E501

        return self.api_client.call_api(
            '/search/image', 'POST',
            path_params,
            query_params,
            header_params,
            body=body_params,
            post_params=form_params,
            files=local_var_files,
            response_type='ImageSearchResponse',  # noqa: E501
            auth_settings=auth_settings,
            async_req=local_var_params.get('async_req'),
            _return_http_data_only=local_var_params.get('_return_http_data_only'),  # noqa: E501
            _preload_content=local_var_params.get('_preload_content', True),
            _request_timeout=local_var_params.get('_request_timeout'),
            collection_formats=collection_formats)

    def upload_image_by_url(self, upload_image_by_url_request, **kwargs):  # noqa: E501
        """Uploads an image to AliExpress to allow it to be used in the image search endpoint   # noqa: E501

        This method makes a synchronous HTTP request by default. To make an
        asynchronous HTTP request, please pass async_req=True
        >>> thread = api.upload_image_by_url(upload_image_by_url_request, async_req=True)
        >>> result = thread.get()

        :param async_req bool
        :param UploadImageByUrlRequest upload_image_by_url_request: The upload image by url request body  (required)
        :return: UploadImageResponse
                 If the method is called asynchronously,
                 returns the request thread.
        """
        kwargs['_return_http_data_only'] = True
        if kwargs.get('async_req'):
            return self.upload_image_by_url_with_http_info(upload_image_by_url_request, **kwargs)  # noqa: E501
        else:
            (data) = self.upload_image_by_url_with_http_info(upload_image_by_url_request, **kwargs)  # noqa: E501
            return data

    def upload_image_by_url_with_http_info(self, upload_image_by_url_request, **kwargs):  # noqa: E501
        """Uploads an image to AliExpress to allow it to be used in the image search endpoint   # noqa: E501

        This method makes a synchronous HTTP request by default. To make an
        asynchronous HTTP request, please pass async_req=True
        >>> thread = api.upload_image_by_url_with_http_info(upload_image_by_url_request, async_req=True)
        >>> result = thread.get()

        :param async_req bool
        :param UploadImageByUrlRequest upload_image_by_url_request: The upload image by url request body  (required)
        :return: UploadImageResponse
                 If the method is called asynchronously,
                 returns the request thread.
        """

        local_var_params = locals()

        all_params = ['upload_image_by_url_request']  # noqa: E501
        all_params.append('async_req')
        all_params.append('_return_http_data_only')
        all_params.append('_preload_content')
        all_params.append('_request_timeout')

        for key, val in six.iteritems(local_var_params['kwargs']):
            if key not in all_params:
                raise TypeError(
                    "Got an unexpected keyword argument '%s'"
                    " to method upload_image_by_url" % key
                )
            local_var_params[key] = val
        del local_var_params['kwargs']
        # verify the required parameter 'upload_image_by_url_request' is set
        if ('upload_image_by_url_request' not in local_var_params or
                local_var_params['upload_image_by_url_request'] is None):
            raise ValueError("Missing the required parameter `upload_image_by_url_request` when calling `upload_image_by_url`")  # noqa: E501

        collection_formats = {}

        path_params = {}

        query_params = []

        header_params = {}

        form_params = []
        local_var_files = {}

        body_params = None
        if 'upload_image_by_url_request' in local_var_params:
            body_params = local_var_params['upload_image_by_url_request']
        # HTTP header `Accept`
        header_params['Accept'] = self.api_client.select_header_accept(
            ['application/json'])  # noqa: E501

        # HTTP header `Content-Type`
        header_params['Content-Type'] = self.api_client.select_header_content_type(  # noqa: E501
            ['application/json'])  # noqa: E501

        # Authentication setting
        auth_settings = ['ApiKeyAuth']  # noqa: E501

        return self.api_client.call_api(
            '/search/image/upload', 'POST',
            path_params,
            query_params,
            header_params,
            body=body_params,
            post_params=form_params,
            files=local_var_files,
            response_type='UploadImageResponse',  # noqa: E501
            auth_settings=auth_settings,
            async_req=local_var_params.get('async_req'),
            _return_http_data_only=local_var_params.get('_return_http_data_only'),  # noqa: E501
            _preload_content=local_var_params.get('_preload_content', True),
            _request_timeout=local_var_params.get('_request_timeout'),
            collection_formats=collection_formats)