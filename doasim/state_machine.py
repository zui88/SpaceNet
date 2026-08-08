from dataclasses import dataclass, field
from enum import Enum, auto

from Utils.stm import StateMachine


class State(Enum):
    ESTIMATION = auto()
    SIMULATION = auto()
    BENCHMARK = auto()

    CLASSICAL_MUSIC = auto()
    ROOT_MUSIC = auto()
    DA_CLASSICAL_MUSIC = auto()
    DA_ROOT_MUSIC = auto()

    DOA = auto()
    DELAY_DOPPLER = auto()


class Event(Enum):
    EST = auto()
    SIM = auto()
    BM = auto()

    CM = auto()
    RM = auto()
    DACM = auto()
    DARM = auto()

    DOA = auto()
    DD = auto()


@dataclass
class PaymentCtx:
    payment_id: str
    audit: list[str] = field(default_factory=list[str])


# Create an instance: this is "the machine"
pay_sm: StateMachine[State, Event, PaymentCtx] = StateMachine()


@pay_sm.transition(State.NEW, Event.AUTHORIZE, State.AUTHORIZED)
def authorize(ctx: PaymentCtx) -> None:
    ctx.audit.append(f"{ctx.payment_id}: authorized")


@pay_sm.transition((State.NEW, State.AUTHORIZED), Event.FAIL, State.FAILED)
def fail(ctx: PaymentCtx) -> None:
    ctx.audit.append(f"{ctx.payment_id}: failed")


@pay_sm.transition(State.AUTHORIZED, Event.CAPTURE, State.CAPTURED)
def capture(ctx: PaymentCtx) -> None:
    ctx.audit.append(f"{ctx.payment_id}: captured")


@pay_sm.transition((State.AUTHORIZED, State.CAPTURED), Event.REFUND, State.REFUNDED)
def refund(ctx: PaymentCtx) -> None:
    ctx.audit.append(f"{ctx.payment_id}: refunded")


@dataclass
class Payment:
    ctx: PaymentCtx
    state: State = State.NEW

    def handle(self, event: Event) -> None:
        self.state = pay_sm.handle(self.ctx, self.state, event)
