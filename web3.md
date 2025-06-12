.. _quickstart:

Quickstart
==========

.. contents:: :local:

.. NOTE:: All code starting with a ``$`` is meant to run on your terminal.
    All code starting with a ``>>>`` is meant to run in a python interpreter,
    like `ipython <https://pypi.org/project/ipython/>`_.

Installation
------------

web3.py can be installed (preferably in a :ref:`virtualenv <setup_environment>`)
using ``pip`` as follows:

.. code-block:: shell

   $ pip install web3


.. NOTE:: If you run into problems during installation, you might have a
    broken environment. See the troubleshooting guide to :ref:`setting up a
    clean environment <setup_environment>`.


Using Web3
----------

This library depends on a connection to an Ethereum node. We call these connections
*Providers* and there are several ways to configure them. The full details can be found
in the :ref:`Providers<providers>` documentation. This Quickstart guide will highlight
a couple of the most common use cases.


Test Provider
*************

If you're just learning the ropes or doing some quick prototyping, you can use a test
provider, `eth-tester <https://github.com/ethereum/eth-tester>`_. This provider includes
some accounts prepopulated with test ether and instantly includes each transaction into a block.
web3.py makes this test provider available via ``EthereumTesterProvider``.

.. note::

  The ``EthereumTesterProvider`` requires additional dependencies. Install them via
  ``pip install "web3[tester]"``, then import and instantiate the provider as seen below.

.. code-block:: python

   >>> from web3 import Web3, EthereumTesterProvider
   >>> w3 = Web3(EthereumTesterProvider())
   >>> w3.is_connected()
   True


Local Providers
***************

The hardware requirements are `steep <https://ethereum.org/en/developers/docs/nodes-and-clients/run-a-node/#top>`_,
but the safest way to interact with Ethereum is to run an Ethereum client on your own hardware.
For locally run nodes, an IPC connection is the most secure option, but HTTP and
websocket configurations are also available. By default, the popular `Geth client <https://geth.ethereum.org/>`_
exposes port ``8545`` to serve HTTP requests and ``8546`` for websocket requests. Connecting
to this local node can be done as follows:

.. code-block:: python

   >>> from web3 import Web3, AsyncWeb3

   # IPCProvider:
   >>> w3 = Web3(Web3.IPCProvider('./path/to/filename.ipc'))
   >>> w3.is_connected()
   True

   # HTTPProvider:
   >>> w3 = Web3(Web3.HTTPProvider('http://127.0.0.1:8545'))
   >>> w3.is_connected()
   True

   # AsyncHTTPProvider:
   >>> w3 = AsyncWeb3(AsyncWeb3.AsyncHTTPProvider('http://127.0.0.1:8545'))
   >>> await w3.is_connected()
   True

   # -- Persistent Connection Providers -- #

   # WebSocketProvider:
   >>> w3 = await AsyncWeb3(AsyncWeb3.WebSocketProvider('ws://127.0.0.1:8546'))
   >>> await w3.is_connected()
   True

   # AsyncIPCProvider:
   >>> w3 = await AsyncWeb3(AsyncWeb3.AsyncIPCProvider('./path/to/filename.ipc'))
   >>> await w3.is_connected()
   True


Remote Providers
****************

The quickest way to interact with the Ethereum blockchain is to use a `remote node provider <https://ethereum.org/en/developers/docs/nodes-and-clients/nodes-as-a-service/#popular-node-services>`_.
You can connect to a remote node by specifying the endpoint, just like the previous local node example:

.. code-block:: python

   >>> from web3 import Web3, AsyncWeb3

   >>> w3 = Web3(Web3.HTTPProvider('https://<your-provider-url>'))

   >>> w3 = AsyncWeb3(AsyncWeb3.AsyncHTTPProvider('https://<your-provider-url>'))

   >>> w3 = await AsyncWeb3(AsyncWeb3.WebSocketProvider('wss://<your-provider-url>'))

This endpoint is provided by the remote node service, typically after you create an account.

.. _first_w3_use:


Getting Blockchain Info
-----------------------

It's time to start using web3.py! Once properly configured, the ``w3`` instance will allow you
to interact with the Ethereum blockchain. Try getting all the information about the latest block:

.. code-block:: python

    >>> w3.eth.get_block('latest')
    {'difficulty': 1,
     'gasLimit': 6283185,
     'gasUsed': 0,
     'hash': HexBytes('0x53b983fe73e16f6ed8178f6c0e0b91f23dc9dad4cb30d0831f178291ffeb8750'),
     'logsBloom': HexBytes('0x00000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000'),
     'miner': '0x0000000000000000000000000000000000000000',
     'mixHash': HexBytes('0x0000000000000000000000000000000000000000000000000000000000000000'),
     'nonce': HexBytes('0x0000000000000000'),
     'number': 0,
     'parentHash': HexBytes('0x0000000000000000000000000000000000000000000000000000000000000000'),
     'proofOfAuthorityData': HexBytes('0x0000000000000000000000000000000000000000000000000000000000000000dddc391ab2bf6701c74d0c8698c2e13355b2e4150000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000'),
     'receiptsRoot': HexBytes('0x56e81f171bcc55a6ff8345e692c0f86e5b48e01b996cadc001622fb5e363b421'),
     'sha3Uncles': HexBytes('0x1dcc4de8dec75d7aab85b567b6ccd41ad312451b948a7413f0a142fd40d49347'),
     'size': 622,
     'stateRoot': HexBytes('0x1f5e460eb84dc0606ab74189dbcfe617300549f8f4778c3c9081c119b5b5d1c1'),
     'timestamp': 0,
     'totalDifficulty': 1,
     'transactions': [],
     'transactionsRoot': HexBytes('0x56e81f171bcc55a6ff8345e692c0f86e5b48e01b996cadc001622fb5e363b421'),
     'uncles': []}

web3.py can help you read block data, sign and send transactions, deploy and interact with contracts,
and a number of other features.

A few suggestions from here:

- The :doc:`overview` page provides a summary of web3.py's features.
- The :class:`w3.eth <web3.eth.Eth>` API contains the most frequently used methods.
- A guide to :ref:`contracts` includes deployment and usage examples.
- The nuances of :doc:`transactions` are explained in another guide.

.. NOTE:: It is recommended that your development environment have the ``PYTHONWARNINGS=default``
    environment variable set. Some deprecation warnings will not show up
    without this variable being set.


.. _overview:

Overview
========

The purpose of this page is to give you a sense of everything web3.py can do
and to serve as a quick reference guide. You'll find a summary of each feature
with links to learn more.

Configuration
-------------

After installing web3.py (via ``pip install web3``), you'll need to configure
a provider endpoint and any middleware you want to use beyond the defaults.


Providers
~~~~~~~~~

:doc:`providers` are how web3.py connects to a blockchain. The library comes with the
following built-in providers:

- :class:`~web3.providers.rpc.HTTPProvider` for connecting to http and https based JSON-RPC servers.
- :class:`~web3.providers.ipc.IPCProvider` for connecting to ipc socket based JSON-RPC servers.
- :class:`~web3.providers.legacy_websocket.LegacyWebSocketProvider` (deprecated) for connecting to websocket based JSON-RPC servers.
- :class:`~web3.providers.async_rpc.AsyncHTTPProvider` for connecting to http and https based JSON-RPC servers asynchronously.
- :class:`~web3.providers.persistent.AsyncIPCProvider` for connecting to ipc socket based JSON-RPC servers asynchronously via a persistent connection.
- :class:`~web3.providers.persistent.WebSocketProvider` for connecting to websocket based JSON-RPC servers asynchronously via a persistent connection.

Examples
````````

.. code-block:: python

   >>> from web3 import Web3, AsyncWeb3

   # IPCProvider:
   >>> w3 = Web3(Web3.IPCProvider('./path/to/filename.ipc'))
   >>> w3.is_connected()
   True

   # HTTPProvider:
   >>> w3 = Web3(Web3.HTTPProvider('http://127.0.0.1:8545'))
   >>> w3.is_connected()
   True

   # AsyncHTTPProvider:
   >>> w3 = AsyncWeb3(AsyncWeb3.AsyncHTTPProvider('http://127.0.0.1:8545'))
   >>> await w3.is_connected()
   True

   # -- Persistent Connection Providers -- #

   # WebSocketProvider:
   >>> w3 = await AsyncWeb3(AsyncWeb3.WebSocketProvider('ws://127.0.0.1:8546'))
   >>> await w3.is_connected()
   True

   # AsyncIPCProvider
   >>> w3 = await AsyncWeb3(AsyncWeb3.AsyncIPCProvider('./path/to/filename.ipc'))
   >>> await w3.is_connected()
   True


For more context, see the :doc:`providers` documentation.


Middleware
~~~~~~~~~~

Your web3.py instance may be further configured via :doc:`middleware`.

web3.py middleware is described using an onion metaphor, where each layer of
middleware may affect both the incoming request and outgoing response from your
provider. The documentation includes a :ref:`visualization <Modifying_Middleware>`
of this idea.

Several middleware are :ref:`included by default <default_middleware>`. You may add to
(:meth:`add <Web3.middleware_onion.add>`, :meth:`inject <Web3.middleware_onion.inject>`,
:meth:`replace <Web3.middleware_onion.replace>`) or disable
(:meth:`remove <Web3.middleware_onion.remove>`,
:meth:`clear <Web3.middleware_onion.clear>`) any of these middleware.


Accounts and Private Keys
-------------------------

Private keys are required to approve any transaction made on your behalf. The manner in
which your key is secured will determine how you create and send transactions in web3.py.

A local node, like `Geth <https://geth.ethereum.org/>`_, may manage your keys for you.
You can reference those keys using the :attr:`web3.eth.accounts <web3.eth.Eth.accounts>`
property.

A hosted node, like `Infura <https://infura.io/>`_, will have no knowledge of your keys.
In this case, you'll need to have your private key available locally for signing
transactions.

Full documentation on the distinction between keys can be found :ref:`here <eth-account>`.
The separate guide to :doc:`transactions` may also help clarify how to manage keys.


Base API
--------

The :ref:`Web3 <web3_base>` class includes a number of convenient utility functions:


Encoding and Decoding Helpers
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

- :meth:`Web3.is_encodable() <web3.w3.is_encodable>`
- :meth:`Web3.to_bytes() <web3.Web3.to_bytes>`
- :meth:`Web3.to_hex() <web3.Web3.to_hex>`
- :meth:`Web3.to_int() <web3.Web3.to_int>`
- :meth:`Web3.to_json() <web3.Web3.to_json>`
- :meth:`Web3.to_text() <web3.Web3.to_text>`


Address Helpers
~~~~~~~~~~~~~~~

- :meth:`Web3.is_address() <web3.Web3.is_address>`
- :meth:`Web3.is_checksum_address() <web3.Web3.is_checksum_address>`
- :meth:`Web3.to_checksum_address() <web3.Web3.to_checksum_address>`


Currency Conversions
~~~~~~~~~~~~~~~~~~~~

- :meth:`Web3.from_wei() <web3.Web3.from_wei>`
- :meth:`Web3.to_wei() <web3.Web3.to_wei>`


Cryptographic Hashing
~~~~~~~~~~~~~~~~~~~~~

- :meth:`Web3.keccak() <web3.Web3.keccak>`
- :meth:`Web3.solidity_keccak() <web3.Web3.solidity_keccak>`


web3.eth API
------------

The most commonly used APIs for interacting with Ethereum can be found under the
:ref:`web3-eth` namespace.


Fetching Data
~~~~~~~~~~~~~

Viewing account balances (:meth:`get_balance <web3.eth.Eth.get_balance>`), transactions
(:meth:`get_transaction <web3.eth.Eth.get_transaction>`), and block data
(:meth:`get_block <web3.eth.Eth.get_block>`) are some of the most common starting
points in web3.py.


API
```

- :meth:`web3.eth.get_balance() <web3.eth.Eth.get_balance>`
- :meth:`web3.eth.get_block() <web3.eth.Eth.get_block>`
- :meth:`web3.eth.get_block_transaction_count() <web3.eth.Eth.get_block_transaction_count>`
- :meth:`web3.eth.get_code() <web3.eth.Eth.get_code>`
- :meth:`web3.eth.get_proof() <web3.eth.Eth.get_proof>`
- :meth:`web3.eth.get_storage_at() <web3.eth.Eth.get_storage_at>`
- :meth:`web3.eth.get_transaction() <web3.eth.Eth.get_transaction>`
- :meth:`web3.eth.get_transaction_by_block() <web3.eth.Eth.get_transaction_by_block>`
- :meth:`web3.eth.get_transaction_count() <web3.eth.Eth.get_transaction_count>`
- :meth:`web3.eth.get_uncle_by_block() <web3.eth.Eth.get_uncle_by_block>`
- :meth:`web3.eth.get_uncle_count() <web3.eth.Eth.get_uncle_count>`


Sending Transactions
~~~~~~~~~~~~~~~~~~~~

The most common use cases will be satisfied with
:meth:`send_transaction <web3.eth.Eth.send_transaction>` or the combination of
:meth:`sign_transaction <web3.eth.Eth.sign_transaction>` and
:meth:`send_raw_transaction <web3.eth.Eth.send_raw_transaction>`. For more context,
see the full guide to :doc:`transactions`.

.. note::

   If interacting with a smart contract, a dedicated API exists. See the next
   section, :ref:`Contracts <overview_contracts>`.


API
```

- :meth:`web3.eth.send_transaction() <web3.eth.Eth.send_transaction>`
- :meth:`web3.eth.sign_transaction() <web3.eth.Eth.sign_transaction>`
- :meth:`web3.eth.send_raw_transaction() <web3.eth.Eth.send_raw_transaction>`
- :meth:`web3.eth.replace_transaction() <web3.eth.Eth.replace_transaction>`
- :meth:`web3.eth.modify_transaction() <web3.eth.Eth.modify_transaction>`
- :meth:`web3.eth.wait_for_transaction_receipt() <web3.eth.Eth.wait_for_transaction_receipt>`
- :meth:`web3.eth.get_transaction_receipt() <web3.eth.Eth.get_transaction_receipt>`
- :meth:`web3.eth.sign() <web3.eth.Eth.sign>`
- :meth:`web3.eth.sign_typed_data() <web3.eth.Eth.sign_typed_data>`
- :meth:`web3.eth.estimate_gas() <web3.eth.Eth.estimate_gas>`
- :meth:`web3.eth.generate_gas_price() <web3.eth.Eth.generate_gas_price>`
- :meth:`web3.eth.set_gas_price_strategy() <web3.eth.Eth.set_gas_price_strategy>`


.. _overview_contracts:

Contracts
---------

web3.py can help you deploy, read from, or execute functions on a deployed contract.

Deployment requires that the contract already be compiled, with its bytecode and ABI
available. This compilation step can be done within
`Remix <http://remix.ethereum.org/>`_ or one of the many contract development
frameworks, such as `Ape <https://docs.apeworx.io/ape/stable/index.html>`_.

Once the contract object is instantiated, calling ``transact`` on the
:meth:`constructor <web3.contract.Contract.constructor>` method will deploy an
instance of the contract:

.. code-block:: python

   >>> ExampleContract = w3.eth.contract(abi=abi, bytecode=bytecode)
   >>> tx_hash = ExampleContract.constructor().transact()
   >>> tx_receipt = w3.eth.wait_for_transaction_receipt(tx_hash)
   >>> tx_receipt.contractAddress
   '0x8a22225eD7eD460D7ee3842bce2402B9deaD23D3'

Once a deployed contract is loaded into a Contract object, the functions of that
contract are available on the ``functions`` namespace:

.. code-block:: python

   >>> deployed_contract = w3.eth.contract(address=tx_receipt.contractAddress, abi=abi)
   >>> deployed_contract.functions.myFunction(42).transact()

If you want to read data from a contract (or see the result of transaction locally,
without executing it on the network), you can use the
:meth:`ContractFunction.call <web3.contract.ContractFunction.call>` method, or the
more concise :attr:`ContractCaller <web3.contract.ContractCaller>` syntax:

.. code-block:: python

   # Using ContractFunction.call
   >>> deployed_contract.functions.getMyValue().call()
   42

   # Using ContractCaller
   >>> deployed_contract.caller().getMyValue()
   42

For more, see the full :ref:`Contracts` documentation.


API
~~~

- :meth:`web3.eth.contract() <web3.eth.Eth.contract>`
- :attr:`Contract.address <web3.contract.Contract.address>`
- :attr:`Contract.abi <web3.contract.Contract.abi>`
- :attr:`Contract.bytecode <web3.contract.Contract.bytecode>`
- :attr:`Contract.bytecode_runtime <web3.contract.Contract.bytecode_runtime>`
- :attr:`Contract.functions <web3.contract.Contract.functions>`
- :attr:`Contract.events <web3.contract.Contract.events>`
- :attr:`Contract.fallback <web3.contract.Contract.fallback.call>`
- :meth:`Contract.constructor() <web3.contract.Contract.constructor>`
- :meth:`Contract.encode_abi() <web3.contract.Contract.encode_abi>`
- :attr:`web3.contract.ContractFunction <web3.contract.ContractFunction>`
- :attr:`web3.contract.ContractEvents <web3.contract.ContractEvents>`


Events, Logs, and Filters
-------------------------

If you want to react to new blocks being mined or specific events being emitted by
a contract, you can leverage ``get_logs``, subscriptions, or filters.

See the :doc:`filters` guide for more information.


API
~~~

- :meth:`web3.eth.subscribe() <web3.eth.Eth.subscribe>`
- :meth:`web3.eth.filter() <web3.eth.Eth.filter>`
- :meth:`web3.eth.get_filter_changes() <web3.eth.Eth.get_filter_changes>`
- :meth:`web3.eth.get_filter_logs() <web3.eth.Eth.get_filter_logs>`
- :meth:`web3.eth.uninstall_filter() <web3.eth.Eth.uninstall_filter>`
- :meth:`web3.eth.get_logs() <web3.eth.Eth.get_logs>`
- :meth:`Contract.events.your_event_name.create_filter() <web3.contract.Contract.events.your_event_name.create_filter>`
- :meth:`Contract.events.your_event_name.build_filter() <web3.contract.Contract.events.your_event_name.build_filter>`
- :meth:`Filter.get_new_entries() <web3.utils.filters.Filter.get_new_entries>`
- :meth:`Filter.get_all_entries() <web3.utils.filters.Filter.get_all_entries>`
- :meth:`Filter.format_entry() <web3.utils.filters.Filter.format_entry>`
- :meth:`Filter.is_valid_entry() <web3.utils.filters.Filter.is_valid_entry>`


Net API
-------

Some basic network properties are available on the ``web3.net`` object:

- :attr:`web3.net.listening`
- :attr:`web3.net.peer_count`
- :attr:`web3.net.version`


ENS
---

`Ethereum Name Service (ENS) <https://ens.domains/>`_ provides the infrastructure
for human-readable addresses. If an address is registered with the ENS registry,
the domain name can be used in place of the address itself. For example, the registered domain
name ``ethereum.eth`` will resolve to the address
``0xde0B295669a9FD93d5F28D9Ec85E40f4cb697BAe``. web3.py has support for ENS, documented
:ref:`here <ens_overview>`.


.. _providers:

Providers
=========

Using Ethereum requires access to an Ethereum node. If you have the means, you're
encouraged to `run your own node`_. (Note that you do not need to stake ether to
run a node.) If you're unable to run your own node, you can use a `remote node`_.

Once you have access to a node, you can connect to it using a **provider**.
Providers generate `JSON-RPC`_ requests and return the response. This is done by submitting
the request to an HTTP, WebSocket, or IPC socket-based server.

.. note::

   web3.py supports one provider per instance. If you have an advanced use case
   that requires multiple providers, create and configure a new web3 instance
   per connection.

If you are already happily connected to your Ethereum node, then you
can skip the rest of this providers section.

.. _run your own node: https://ethereum.org/en/developers/docs/nodes-and-clients/run-a-node/
.. _remote node: https://ethereum.org/en/developers/docs/nodes-and-clients/nodes-as-a-service/
.. _JSON-RPC: https://ethereum.org/en/developers/docs/apis/json-rpc/

.. _choosing_provider:

Choosing a Provider
-------------------

Most nodes have a variety of ways to connect to them. Most commonly:

1. IPC (uses local filesystem: fastest and most secure)
2. WebSocket (works remotely, faster than HTTP)
3. HTTP (more nodes support it)

If you're not sure how to decide, choose this way:

- If you have the option of running web3.py on the same machine as the node, choose IPC.
- If you must connect to a node on a different computer, use WebSocket.
- If your node does not support WebSocket, use HTTP.

Once you have decided how to connect, you'll select and configure the appropriate provider
class:

- :class:`~web3.providers.rpc.HTTPProvider`
- :class:`~web3.providers.ipc.IPCProvider`
- :class:`~web3.providers.async_rpc.AsyncHTTPProvider`
- :class:`~web3.providers.persistent.AsyncIPCProvider` (Persistent Connection Provider)
- :class:`~web3.providers.persistent.WebSocketProvider` (Persistent Connection Provider)

Each provider above links to the documentation on how to properly initialize that
provider. Once you have reviewed the relevant documentation for the provider of your
choice, you are ready to :ref:`get started with web3.py<first_w3_use>`.

Provider via Environment Variable
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Alternatively, you can set the environment variable ``WEB3_PROVIDER_URI``
before starting your script, and web3 will look for that provider first.

Valid formats for this environment variable are:

- ``file:///path/to/node/rpc-json/file.ipc``
- ``http://192.168.1.2:8545``
- ``https://node.ontheweb.com``
- ``ws://127.0.0.1:8546``


Auto-initialization Provider Shortcuts
--------------------------------------

Geth dev Proof of Authority
~~~~~~~~~~~~~~~~~~~~~~~~~~~

To connect to a ``geth --dev`` Proof of Authority instance with
the POA middleware loaded by default:

.. code-block:: python

    >>> from web3.auto.gethdev import w3

    # confirm that the connection succeeded
    >>> w3.is_connected()
    True

Or, connect to an async web3 instance:

.. code-block:: python

    >>> from web3.auto.gethdev import async_w3
    >>> await async_w3.provider.connect()

    # confirm that the connection succeeded
    >>> await async_w3.is_connected()
    True


Built In Providers
------------------

Web3 ships with the following providers which are appropriate for connecting to
local and remote JSON-RPC servers.


HTTPProvider
~~~~~~~~~~~~

.. py:class:: web3.providers.rpc.HTTPProvider(endpoint_uri, request_kwargs={}, session=None, exception_retry_configuration=ExceptionRetryConfiguration())

    This provider handles interactions with an HTTP or HTTPS based JSON-RPC server.

    * ``endpoint_uri`` should be the full URI to the RPC endpoint such as
      ``'https://localhost:8545'``.  For RPC servers behind HTTP connections
      running on port 80 and HTTPS connections running on port 443 the port can
      be omitted from the URI.
    * ``request_kwargs`` should be a dictionary of keyword arguments which
      will be passed onto each http/https POST request made to your node.
    * ``session`` allows you to pass a ``requests.Session`` object initialized
      as desired.
    * ``exception_retry_configuration`` is an instance of the
      :class:`~web3.providers.rpc.utils.ExceptionRetryConfiguration`
      class which allows you to configure how the provider should handle exceptions
      when making certain requests. Setting this to ``None`` will disable
      exception retries.

    .. code-block:: python

        >>> from web3 import Web3
        >>> w3 = Web3(Web3.HTTPProvider("http://127.0.0.1:8545"))

    Note that you should create only one HTTPProvider with the same provider URL
    per python process, as the HTTPProvider recycles underlying TCP/IP
    network connections, for better performance. Multiple HTTPProviders with different
    URLs will work as expected.

    Under the hood, the ``HTTPProvider`` uses the python requests library for
    making requests.  If you would like to modify how requests are made, you can
    use the ``request_kwargs`` to do so.  A common use case for this is increasing
    the timeout for each request.


    .. code-block:: python

        >>> from web3 import Web3
        >>> w3 = Web3(Web3.HTTPProvider("http://127.0.0.1:8545", request_kwargs={'timeout': 60}))


    To tune the connection pool size, you can pass your own ``requests.Session``.

    .. code-block:: python

        >>> from web3 import Web3
        >>> adapter = requests.adapters.HTTPAdapter(pool_connections=20, pool_maxsize=20)
        >>> session = requests.Session()
        >>> session.mount('http://', adapter)
        >>> session.mount('https://', adapter)
        >>> w3 = Web3(Web3.HTTPProvider("http://127.0.0.1:8545", session=session))


IPCProvider
~~~~~~~~~~~

.. py:class:: web3.providers.ipc.IPCProvider(ipc_path=None, timeout=10)

    This provider handles interaction with an IPC Socket based JSON-RPC
    server.

    *  ``ipc_path`` is the filesystem path to the IPC socket:

    .. code-block:: python

        >>> from web3 import Web3
        >>> w3 = Web3(Web3.IPCProvider("~/Library/Ethereum/geth.ipc"))

    If no ``ipc_path`` is specified, it will use a default depending on your operating
    system.

    - On Linux and FreeBSD: ``~/.ethereum/geth.ipc``
    - On Mac OS: ``~/Library/Ethereum/geth.ipc``
    - On Windows: ``\\.\pipe\geth.ipc``


AsyncHTTPProvider
~~~~~~~~~~~~~~~~~

.. py:class:: web3.providers.rpc.AsyncHTTPProvider(endpoint_uri, request_kwargs={}, exception_retry_configuration=ExceptionRetryConfiguration())

    This provider handles interactions with an HTTP or HTTPS based JSON-RPC server asynchronously.

    * ``endpoint_uri`` should be the full URI to the RPC endpoint such as
      ``'https://localhost:8545'``.  For RPC servers behind HTTP connections
      running on port 80 and HTTPS connections running on port 443 the port can
      be omitted from the URI.
    * ``request_kwargs`` should be a dictionary of keyword arguments which
      will be passed onto each http/https POST request made to your node.
    * ``exception_retry_configuration`` is an instance of the
      :class:`~web3.providers.rpc.utils.ExceptionRetryConfiguration`
      class which allows you to configure how the provider should handle exceptions
      when making certain requests. Setting this to ``None`` will disable
      exception retries.

    The ``cache_async_session()`` method allows you to use your own
    ``aiohttp.ClientSession`` object.

    .. code-block:: python

        >>> from aiohttp import ClientSession
        >>> from web3 import AsyncWeb3, AsyncHTTPProvider

        >>> w3 = AsyncWeb3(AsyncHTTPProvider(endpoint_uri))

        >>> # If you want to pass in your own session:
        >>> custom_session = ClientSession()
        >>> await w3.provider.cache_async_session(custom_session) # This method is an async method so it needs to be handled accordingly
        >>> # when you're finished, disconnect:
        >>> w3.provider.disconnect()

    Under the hood, the ``AsyncHTTPProvider`` uses the python
    `aiohttp <https://docs.aiohttp.org/en/stable/>`_ library for making requests.

Persistent Connection Providers
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Persistent Connection Base Class
++++++++++++++++++++++++++++++++

.. note::
    This class is not meant to be used directly. If your provider class inherits
    from this class, look to these docs for additional configuration options.

.. py:class:: web3.providers.persistent.PersistentConnectionProvider(\
        request_timeout: float = 50.0, \
        subscription_response_queue_size: int = 500, \
        silence_listener_task_exceptions: bool = False \
        max_connection_retries: int = 5, \
        request_information_cache_size: int = 500, \
    )

    This is a base provider class, inherited by the following providers:

        - :class:`~web3.providers.persistent.WebSocketProvider`
        - :class:`~web3.providers.persistent.AsyncIPCProvider`

    It handles interactions with a persistent connection to a JSON-RPC server. Among
    its configuration, it houses all of the
    :class:`~web3.providers.persistent.request_processor.RequestProcessor` logic for
    handling the asynchronous sending and receiving of requests and responses. See
    the :ref:`internals__persistent_connection_providers` section for more details on
    the internals of persistent connection providers.

    * ``request_timeout`` is the timeout in seconds, used when sending data over the
      connection and waiting for a response to be received from the listener task.
      Defaults to ``50.0``.

    * ``subscription_response_queue_size`` is the size of the queue used to store
      subscription responses, defaults to ``500``. While messages are being consumed,
      this queue should never fill up as it is a transient queue and meant to handle
      asynchronous receiving and processing of responses. When in sync with the
      socket stream, this queue should only ever store 1 to a few messages at a time.

    * ``silence_listener_task_exceptions`` is a boolean that determines whether
      exceptions raised by the listener task are silenced. Defaults to ``False``,
      raising any exceptions that occur in the listener task.

    * ``max_connection_retries`` is the maximum number of times to retry a connection
      to the provider when initializing the provider. Defaults to ``5``.

    * ``request_information_cache_size`` specifies the size of the transient cache for
      storing request details, enabling the provider to process responses based on the
      original request information. Defaults to ``500``.

AsyncIPCProvider
++++++++++++++++

.. py:class:: web3.providers.persistent.AsyncIPCProvider(ipc_path=None, max_connection_retries=5)

    This provider handles asynchronous, persistent interaction with an IPC Socket based
    JSON-RPC server.

    *  ``ipc_path`` is the filesystem path to the IPC socket:
    *  ``read_buffer_limit`` is the maximum size of data, in bytes, that can be read
       from the socket at one time. Defaults to 20MB (20 * 1024 * 1024). Raises
       ``ReadBufferLimitReached`` if the limit is reached, suggesting that the buffer
       limit be increased.

    This provider inherits from the
    :class:`~web3.providers.persistent.PersistentConnectionProvider` class. Refer to
    the :class:`~web3.providers.persistent.PersistentConnectionProvider` documentation
    for details on additional configuration options available for this provider.

    If no ``ipc_path`` is specified, it will use a default depending on your operating
    system.

    - On Linux and FreeBSD: ``~/.ethereum/geth.ipc``
    - On Mac OS: ``~/Library/Ethereum/geth.ipc``
    - On Windows: ``\\.\pipe\geth.ipc``

WebSocketProvider
+++++++++++++++++

.. py:class:: web3.providers.persistent.WebSocketProvider(endpoint_uri: str, websocket_kwargs: Dict[str, Any] = {}, use_text_frames: bool = False)

    This provider handles interactions with an WS or WSS based JSON-RPC server.

    * ``endpoint_uri`` should be the full URI to the RPC endpoint such as
      ``'ws://localhost:8546'``.
    * ``websocket_kwargs`` this should be a dictionary of keyword arguments which
      will be passed onto the ws/wss websocket connection.
    * ``use_text_frames`` will ensure websocket data is sent as text frames
      for servers that do not support binary communication.

    This provider inherits from the
    :class:`~web3.providers.persistent.PersistentConnectionProvider` class. Refer to
    the :class:`~web3.providers.persistent.PersistentConnectionProvider` documentation
    for details on additional configuration options available for this provider.

    Under the hood, the ``WebSocketProvider`` uses the python websockets library for
    making requests.  If you would like to modify how requests are made, you can
    use the ``websocket_kwargs`` to do so.  See the `websockets documentation`_ for
    available arguments.


.. _subscription-examples:

Using Persistent Connection Providers
+++++++++++++++++++++++++++++++++++++

The ``AsyncWeb3`` class may be used as a context manager, utilizing the ``async with``
syntax, when instantiating with a
:class:`~web3.providers.persistent.PersistentConnectionProvider`. This will
automatically close the connection when the context manager exits and is the
recommended way to initiate a persistent connection to the provider.

A similar example using a ``websockets`` connection as an asynchronous context manager
can be found in the `websockets connection`_ docs.

.. code-block:: python

        >>> import asyncio
        >>> from web3 import AsyncWeb3
        >>> from web3.providers.persistent import (
        ...     AsyncIPCProvider,
        ...     WebSocketProvider,
        ... )

        >>> LOG = True  # toggle debug logging
        >>> if LOG:
        ...     import logging
        ...     # logger = logging.getLogger("web3.providers.AsyncIPCProvider")  # for the AsyncIPCProvider
        ...     logger = logging.getLogger("web3.providers.WebSocketProvider")  # for the WebSocketProvider
        ...     logger.setLevel(logging.DEBUG)
        ...     logger.addHandler(logging.StreamHandler())

        >>> async def context_manager_subscription_example():
        ...     #  async with AsyncWeb3(AsyncIPCProvider("./path/to.filename.ipc") as w3:  # for the AsyncIPCProvider
        ...     async with AsyncWeb3(WebSocketProvider(f"ws://127.0.0.1:8546")) as w3:  # for the WebSocketProvider
        ...         # subscribe to new block headers
        ...         subscription_id = await w3.eth.subscribe("newHeads")
        ...
        ...         async for response in w3.socket.process_subscriptions():
        ...             print(f"{response}\n")
        ...             # handle responses here
        ...
        ...             if some_condition:
        ...                 # unsubscribe from new block headers and break out of
        ...                 # iterator
        ...                 await w3.eth.unsubscribe(subscription_id)
        ...                 break
        ...
        ...         # still an open connection, make any other requests and get
        ...         # responses via send / receive
        ...         latest_block = await w3.eth.get_block("latest")
        ...         print(f"Latest block: {latest_block}")
        ...
        ...         # the connection closes automatically when exiting the context
        ...         # manager (the `async with` block)

        >>> asyncio.run(context_manager_subscription_example())


The ``AsyncWeb3`` class may also be used as an asynchronous iterator, utilizing the
``async for`` syntax, when instantiating with a
:class:`~web3.providers.persistent.PersistentConnectionProvider`. This may be used to
set up an indefinite websocket connection and reconnect automatically if the connection
is lost.

A similar example using a ``websockets`` connection as an asynchronous iterator can
be found in the `websockets connection`_ docs.

.. _`websockets connection`: https://websockets.readthedocs.io/en/stable/reference/asyncio/client.html#websockets.client.connect

.. code-block:: python

    >>> import asyncio
    >>> import websockets
    >>> from web3 import AsyncWeb3
    >>> from web3.providers.persistent import (
    ...     AsyncIPCProvider,
    ...     WebSocketProvider,
    ... )

    >>> async def subscription_iterator_example():
    ...     # async for w3 in AsyncWeb3(AsyncIPCProvider("./path/to/filename.ipc")):  # for the AsyncIPCProvider
    ...     async for w3 in AsyncWeb3(WebSocketProvider(f"ws://127.0.0.1:8546")):  # for the WebSocketProvider
    ...         try:
    ...             ...
    ...         except websockets.ConnectionClosed:
    ...             continue

    # run the example
    >>> asyncio.run(subscription_iterator_example())


Awaiting the instantiation with a
:class:`~web3.providers.persistent.PersistentConnectionProvider`, or instantiating
and awaiting the ``connect()`` method is also possible. Both of these examples are
shown below.

.. code-block:: python

    >>> async def await_instantiation_example():
    ...     # w3 = await AsyncWeb3(AsyncIPCProvider("./path/to/filename.ipc"))  # for the AsyncIPCProvider
    ...     w3 = await AsyncWeb3(WebSocketProvider(f"ws://127.0.0.1:8546"))  # for the WebSocketProvider
    ...
    ...     # some code here
    ...
    ...     # manual cleanup
    ...     await w3.provider.disconnect()

    # run the example
    >>> asyncio.run(await_instantiation_example())

.. code-block:: python

    >>> async def await_provider_connect_example():
    ...     # w3 = AsyncWeb3(AsyncIPCProvider("./path/to/filename.ipc"))  # for the AsyncIPCProvider
    ...     w3 = AsyncWeb3(WebSocketProvider(f"ws://127.0.0.1:8546"))  # for the WebSocketProvider
    ...     await w3.provider.connect()
    ...
    ...     # some code here
    ...
    ...     # manual cleanup
    ...     await w3.provider.disconnect()

    # run the example
    >>> asyncio.run(await_provider_connect_example())

:class:`~web3.providers.persistent.PersistentConnectionProvider` classes use the
:class:`~web3.providers.persistent.request_processor.RequestProcessor` class under the
hood to sync up the receiving of responses and response processing for one-to-one and
one-to-many request-to-response requests. Refer to the
:class:`~web3.providers.persistent.request_processor.RequestProcessor`
documentation for details.

AsyncWeb3 with Persistent Connection Providers
++++++++++++++++++++++++++++++++++++++++++++++

When an ``AsyncWeb3`` class is connected to a
:class:`~web3.providers.persistent.PersistentConnectionProvider`, some attributes and
methods become available.

    .. py:attribute:: socket

        The public API for interacting with the websocket connection is available via
        the ``socket`` attribute of the ``Asyncweb3`` class. This attribute is an
        instance of the
        :class:`~web3.providers.persistent.persistent_connection.PersistentConnection`
        class and is the main interface for interacting with the socket connection.


Interacting with the Persistent Connection
++++++++++++++++++++++++++++++++++++++++++

.. py:class:: web3.providers.persistent.persistent_connection.PersistentConnection

    This class handles interactions with a persistent socket connection. It is available
    via the ``socket`` attribute on the ``AsyncWeb3`` class. The
    ``PersistentConnection`` class has the following methods and attributes:

    .. py:attribute:: subscriptions

        This attribute returns the current active subscriptions as a dict mapping
        the subscription ``id`` to a dict of metadata about the subscription
        request.

    .. py:method:: process_subscriptions()

        This method is available for listening to websocket subscriptions indefinitely.
        It is an asynchronous iterator that yields strictly one-to-many
        (e.g. ``eth_subscription`` responses) request-to-response messages from the
        websocket connection. To receive responses for one-to-one request-to-response
        calls, use the standard API for making requests via the appropriate module
        (e.g. ``block_num = await w3.eth.block_number``)

        The responses from this method are formatted by *web3.py* formatters and run
        through the middleware that were present at the time of subscription.
        Examples on how to use this method can be seen above in the
        `Using Persistent Connection Providers`_ section.

    .. py:method:: send(method: RPCEndpoint, params: Sequence[Any])

        This method is available strictly for sending raw requests to the socket,
        if desired. It is not recommended to use this method directly, as the
        responses will not be formatted by *web3.py* formatters or run through the
        middleware. Instead, use the methods available on the respective web3
        module. For example, use ``w3.eth.get_block("latest")`` instead of
        ``w3.socket.send("eth_getBlockByNumber", ["latest", True])``.

    .. py:method:: recv()

        The ``recv()`` method can be used to receive the next response for a request
        from the socket. The response from this method is the raw response. This is not
        the recommended way to receive a response for a request, as it is not formatted
        by *web3.py* formatters or run through the middleware. Instead, use the methods
        available on the respective web3 module
        (e.g. ``block_num = await w3.eth.block_number``) for retrieving responses for
        one-to-one request-to-response calls.

    .. py:method:: make_request(method: RPCEndpoint, params: Sequence[Any])

        This method is available for making requests to the socket and retrieving the
        response. It is not recommended to use this method directly, as the responses
        will not be properly formatted by *web3.py* formatters or run through the
        middleware. Instead, use the methods available on the respective web3 module.
        For example, use ``w3.eth.get_block("latest")`` instead of
        ``w3.socket.make_request("eth_getBlockByNumber", ["latest", True])``.


LegacyWebSocketProvider
~~~~~~~~~~~~~~~~~~~~~~~

.. warning::

        ``LegacyWebSocketProvider`` is deprecated and is likely to be removed in a
        future major release. Please use ``WebSocketProvider`` instead.

.. py:class:: web3.providers.legacy_websocket.LegacyWebSocketProvider(endpoint_uri[, websocket_timeout, websocket_kwargs])

    This provider handles interactions with an WS or WSS based JSON-RPC server.

    * ``endpoint_uri`` should be the full URI to the RPC endpoint such as
      ``'ws://localhost:8546'``.
    * ``websocket_timeout`` is the timeout in seconds, used when receiving or
      sending data over the connection. Defaults to 10.
    * ``websocket_kwargs`` this should be a dictionary of keyword arguments which
      will be passed onto the ws/wss websocket connection.

    .. code-block:: python

        >>> from web3 import Web3
        >>> w3 = Web3(Web3.LegacyWebSocketProvider("ws://127.0.0.1:8546"))

    Under the hood, ``LegacyWebSocketProvider`` uses the python ``websockets`` library for
    making requests.  If you would like to modify how requests are made, you can
    use the ``websocket_kwargs`` to do so.  See the `websockets documentation`_ for
    available arguments.

    .. _`websockets documentation`: https://websockets.readthedocs.io/en/stable/reference/asyncio/client.html#websockets.client.WebSocketClientProtocol

    Unlike HTTP connections, the timeout for WS connections is controlled by a
    separate ``websocket_timeout`` argument, as shown below.


    .. code-block:: python

        >>> from web3 import Web3
        >>> w3 = Web3(Web3.LegacyWebSocketProvider("ws://127.0.0.1:8546", websocket_timeout=60))


AutoProvider
~~~~~~~~~~~~

:class:`~web3.providers.auto.AutoProvider` is the default used when initializing
:class:`web3.Web3` without any providers. There's rarely a reason to use it
explicitly.

.. py:currentmodule:: web3.providers.eth_tester

EthereumTesterProvider
~~~~~~~~~~~~~~~~~~~~~~

.. warning:: Experimental:  This provider is experimental. There are still significant
    gaps in functionality. However it is being actively developed and supported.

.. py:class:: EthereumTesterProvider(ethereum_tester=None, api_endpoints=None)
.. py:class:: AsyncEthereumTesterProvider(ethereum_tester=None, api_endpoints=None)

    This provider integrates with the ``eth-tester`` library. The ``ethereum_tester``
    constructor argument should be an instance of the :class:`~eth_tester.EthereumTester`
    or a subclass of :class:`~eth_tester.backends.base.BaseChainBackend` class provided
    by the ``eth-tester`` library. The ``api_endpoints`` argument should be a ``dict``
    of RPC endpoints. You can see the structure and defaults `here <https://github.com/ethereum/web3.py/blob/283b536c7d53e605c61468941e3fc07a6c5d0c09/web3/providers/eth_tester/defaults.py#L228>`_.
    If you would like a custom ``eth-tester`` instance to test with, see the
    ``eth-tester`` library `documentation <https://github.com/ethereum/eth-tester>`_
    for details.

    .. code-block:: python

        >>> from web3 import Web3, EthereumTesterProvider
        >>> w3 = Web3(EthereumTesterProvider())

.. NOTE:: To install the needed dependencies to use EthereumTesterProvider, you can
    install the pip extras package that has the correct interoperable versions of the
    ``eth-tester`` and ``py-evm`` dependencies needed: e.g. ``pip install "web3[tester]"``